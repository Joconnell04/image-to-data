import { Clipboard, Toast, showToast, getPreferenceValues, closeMainWindow, showHUD } from "@raycast/api";
import { execSync } from "child_process";
import fs from "fs";
import path from "path";
import os from "os";
import { OpenAI } from "openai";

type Mode = "table" | "chart" | "diagram" | "text" | "general";

type Preferences = {
  openaiApiKey: string;
  routerModel: string;
  extractModel: string;
  maxOutputChars: string;
  includeConfidence?: boolean;
  forceMode: "auto" | Mode;
  debugLogging?: boolean;
};

type RouterResult = {
  type: Mode;
  complexity?: "simple" | "complex";
  format?: string;
  priorities?: string[];
};

// Type for OpenAI Responses API output
interface OpenAIResponseOutput {
  output_text?: string;
  output?: Array<{
    content?: Array<{
      text?: string;
    }>;
  }>;
}

export default async function main() {
  try {
    const prefs = getPreferenceValues<Preferences>();
    const maxChars = parseInt(prefs.maxOutputChars, 10) || 12000;

    // Close Raycast window first so it doesn't appear in screenshot
    await closeMainWindow();

    const imagePath = await captureScreenshot();
    if (!imagePath) {
      await showHUD("Screenshot cancelled");
      return;
    }

    await showHUD("Extracting data...");

    const dataUrl = await imageFileToDataUrl(imagePath);
    const client = new OpenAI({ apiKey: prefs.openaiApiKey });

    const mode = await decideMode(client, dataUrl, prefs);
    const raw = await extractFromImage(client, dataUrl, mode, prefs);
    const finalText = sanitizeOutput(raw, maxChars, prefs.includeConfidence);

    await Clipboard.copy(finalText);
    await showToast({
      style: Toast.Style.Success,
      title: "Copied extracted text",
      message: `Mode: ${mode}`,
    });

    // Clean up temp file
    try {
      await fs.promises.unlink(imagePath);
    } catch {
      // ignore cleanup errors
    }
  } catch (error) {
    console.error(error);
    const message = error instanceof Error ? error.message : "Unexpected error";
    await showToast({
      style: Toast.Style.Failure,
      title: "Extraction failed",
      message,
    });
  }
}

async function captureScreenshot(): Promise<string | null> {
  const tmpPath = path.join(os.tmpdir(), `raycast-capture-${Date.now()}.png`);

  try {
    // Use full path and execSync like other Raycast extensions
    execSync(`/usr/sbin/screencapture -i '${tmpPath}'`);

    // Check if file was created (user didn't cancel with Escape)
    if (fs.existsSync(tmpPath)) {
      const stats = fs.statSync(tmpPath);
      if (stats.size > 0) {
        return tmpPath;
      }
    }
  } catch (err) {
    console.error("Screenshot error:", err);
    return null;
  }

  return null;
}

async function imageFileToDataUrl(filePath: string): Promise<string> {
  const buffer = await fs.promises.readFile(filePath);
  const mime = extToMime(path.extname(filePath).toLowerCase()) ?? "image/png";
  const base64 = buffer.toString("base64");
  return `data:${mime};base64,${base64}`;
}

function extToMime(ext: string): string | null {
  switch (ext) {
    case ".png":
      return "image/png";
    case ".jpg":
    case ".jpeg":
      return "image/jpeg";
    case ".webp":
      return "image/webp";
    case ".tif":
    case ".tiff":
      return "image/tiff";
    default:
      return null;
  }
}

async function decideMode(client: OpenAI, dataUrl: string, prefs: Preferences): Promise<Mode> {
  if (prefs.forceMode !== "auto") {
    return prefs.forceMode;
  }

  const systemPrompt = "You are a routing function. Return ONLY valid minified JSON. No prose.";
  const userPrompt = `Classify this image for data extraction. Return JSON with:
- type: "text" (screenshot with mostly text/UI), "table" (structured rows/columns), "chart" (graphs with data series/trends), "diagram" (flowcharts/architecture), or "general" (photos/mixed)
- complexity: "simple" (basic text extraction needed) or "complex" (multiple series, trends, relationships to analyze)

Choose "text" for: app screenshots, documents, code, UI with text, simple lists.
Choose "chart" with complexity "complex" for: scatter plots, multi-series graphs, charts with trends to analyze.
Return: {"type":"...","complexity":"..."}`;

  try {
    const response = await client.responses.create({
      model: prefs.routerModel,
      temperature: 0,
      max_output_tokens: 200,
      input: [
        {
          role: "system",
          content: [{ type: "input_text", text: systemPrompt }],
        },
        {
          role: "user",
          content: [
            { type: "input_text", text: userPrompt },
            { type: "input_image", image_url: dataUrl },
          ],
        },
      ],
    });

    const typedResp = response as unknown as OpenAIResponseOutput;
    const text = typedResp.output_text ?? typedResp.output?.[0]?.content?.[0]?.text ?? "";
    const parsed = safeParseRouter(text);
    if (prefs.debugLogging) {
      console.log("Router raw:", text);
      console.log("Router parsed:", parsed);
    }
    if (
      parsed?.type === "table" ||
      parsed?.type === "chart" ||
      parsed?.type === "diagram" ||
      parsed?.type === "text" ||
      parsed?.type === "general"
    ) {
      return parsed.type;
    }
  } catch (error) {
    if (prefs.debugLogging) {
      console.error("Router error", error);
    }
  }

  return "general";
}

function safeParseRouter(text: string): RouterResult | null {
  try {
    return JSON.parse(text) as RouterResult;
  } catch {
    // Try to extract JSON substring
    const match = text.match(/\{.*\}/s);
    if (match) {
      try {
        return JSON.parse(match[0]) as RouterResult;
      } catch {
        return null;
      }
    }
    return null;
  }
}

async function extractFromImage(client: OpenAI, dataUrl: string, mode: Mode, prefs: Preferences): Promise<string> {
  const globalRules = [
    "Follow the format instructions EXACTLY.",
    "Extract ALL visible text exactly when readable; preserve capitalization and units.",
    "Never invent numbers. If unreadable, output [illegible] or use ~ for approximations.",
    "No markdown code fences.",
  ];

  const includeConfidenceLine = prefs.includeConfidence
    ? "End with a single line: Confidence: high|medium|low (choose based on legibility)."
    : "";

  let modeInstructions = "";
  switch (mode) {
    case "table":
      modeInstructions = `
Output TSV only (tabs between cells, newline between rows).
If header exists, include it as first row.
Preserve blank cells; for merged cells repeat value if clear, else leave blank and add final line 'Notes: merged cells present'.
`;
      break;
    case "chart":
      modeInstructions = `
You MUST analyze this chart and provide per-series trend data. Follow this EXACT format:

TYPE: [scatter/line/bar/pie/etc]
TITLE: [title or "none"]
X-AXIS: [label] range [min to max]
Y-AXIS: [label] range [min to max]
LEGEND: [color1]=name1, [color2]=name2, ...

SERIES ANALYSIS (for EACH series):
- [Series color/name]: center=(~x,~y), spread=[tight/wide/elongated], x=[min,max], y=[min,max]
  Pattern: [describe the shape/behavior - use terms like:]
    * linear increasing/decreasing (note slope: gradual, steep, ~45°)
    * exponential growth/decay
    * logarithmic curve
    * constant/flat/horizontal
    * vertical band/spike
    * parabolic/curved
    * scattered/no clear pattern
  Notable: [any of these if present:]
    * steep drop-off at x=~[value]
    * sharp increase at x=~[value]
    * plateau between x=~[start] and x=~[end]
    * inflection point at (~x, ~y)
    * outliers at (~x, ~y)
    * asymptotic behavior approaching y=~[value]
    * peak/maximum at (~x, ~y)
    * trough/minimum at (~x, ~y)
    * discontinuity/gap at x=~[value]

CROSS-SERIES PATTERNS:
- Correlations: [positive/negative/none between which series]
- Clusters: [where do multiple series overlap or group]
- Separations: [clear boundaries between series]
- Crossover points: [where series intersect, if any]

You MUST describe EVERY series with specific coordinates and pattern characteristics.
`;
      break;
    case "diagram":
      modeInstructions = `
Extract diagram structure concisely:
Components: bullet list of main elements
Connections: A -> B (label) for each relationship
Labels: extract all text exactly
`;
      break;
    case "text":
      modeInstructions = `
Extract all visible text in reading order.
Preserve formatting, line breaks, and hierarchy where apparent.
For UI elements: [Button: "label"] or [Menu: item1, item2]
For code: preserve indentation exactly.
No analysis needed - just accurate text extraction.
`;
      break;
    case "general":
      modeInstructions = `
First, identify what type of content this is.

IF THIS IS A CHART/GRAPH with data series or visual patterns:
Analyze thoroughly with:
- TYPE, AXES with ranges, LEGEND
- For EACH series/color: center, spread, coordinate ranges
- Pattern type: linear, exponential, logarithmic, constant, scattered, etc.
- Notable features: steep drop-offs, sharp increases, plateaus, peaks, troughs, outliers, asymptotes, inflection points, crossovers
- Cross-series correlations and separations

IF THIS IS TEXT/UI/DOCUMENT:
Extract all visible text in reading order.

IF THIS IS A DIAGRAM:
List components and their connections.

IF THIS IS A PHOTO or other:
Brief factual description of key elements.

For data visualizations: describe mathematical patterns and notable features with coordinates.
`;
      break;
  }

  const systemPrompt = [
    "You convert clipboard images into structured text for pasting.",
    ...globalRules,
    includeConfidenceLine,
  ]
    .filter(Boolean)
    .join("\n");

  const userPrompt = `Mode: ${mode.toUpperCase()}\n${modeInstructions}\nReturn the extracted content now.`;

  const response = await client.responses.create({
    model: prefs.extractModel,
    temperature: 0,
    max_output_tokens: 4000,
    input: [
      {
        role: "system",
        content: [{ type: "input_text", text: systemPrompt }],
      },
      {
        role: "user",
        content: [
          { type: "input_text", text: userPrompt },
          { type: "input_image", image_url: dataUrl },
        ],
      },
    ],
  });

  const typedResp = response as unknown as OpenAIResponseOutput;
  const text = typedResp.output_text ?? typedResp.output?.[0]?.content?.[0]?.text ?? "";
  return text;
}

function sanitizeOutput(text: string, maxChars: number, includeConfidence?: boolean): string {
  let out = (text ?? "").trim();
  out = out.replace(/^(this image|the image)\s+(shows|is|contains)\s*[:-]?\s*/i, "");
  if (includeConfidence && !/^\s*confidence:\s*(high|medium|low)\s*$/im.test(out)) {
    out = `${out}\nConfidence: medium`;
  }
  if (out.length > maxChars) {
    out = `${out.slice(0, maxChars)}\n[truncated]`;
  }
  return out.trim();
}
