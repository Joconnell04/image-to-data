import { Clipboard, Toast, showToast, getPreferenceValues, closeMainWindow, showHUD } from "@raycast/api";
import { execSync } from "child_process";
import fs from "fs";
import path from "path";
import os from "os";
import { OpenAI } from "openai";

type Mode = "table" | "chart" | "diagram" | "text" | "general";

type Preferences = {
  openaiApiKey: string;
  extractModel: string;
  maxOutputChars: string;
  includeConfidence?: boolean;
  forceMode: "auto" | Mode;
  debugLogging?: boolean;
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

    const raw = await extractFromImage(client, dataUrl, prefs);
    const finalText = sanitizeOutput(raw, maxChars, prefs.includeConfidence);

    await Clipboard.copy(finalText);
    await showToast({
      style: Toast.Style.Success,
      title: "Copied extracted text",
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

async function extractFromImage(client: OpenAI, dataUrl: string, prefs: Preferences): Promise<string> {
  const globalRules = [
    "Follow the format instructions EXACTLY.",
    "Extract ALL visible text exactly when readable; preserve capitalization and units.",
    "Never invent numbers. If unreadable, output [illegible] or use ~ for approximations.",
    "No markdown code fences.",
  ];

  const includeConfidenceLine = prefs.includeConfidence
    ? "End with a single line: Confidence: high|medium|low (choose based on legibility)."
    : "";

  // Build the prompt based on forceMode
  let userPrompt: string;

  if (prefs.forceMode !== "auto") {
    // Use specific mode instructions when forced
    userPrompt = getForcedModePrompt(prefs.forceMode);
  } else {
    // Use unified auto-detect prompt
    userPrompt = getUnifiedPrompt();
  }

  const systemPrompt = [
    "You convert clipboard images into structured text for pasting.",
    ...globalRules,
    includeConfidenceLine,
  ]
    .filter(Boolean)
    .join("\n");

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

  if (prefs.debugLogging) {
    console.log("Extraction result:", text.substring(0, 500));
  }

  return text;
}

function getForcedModePrompt(mode: Mode): string {
  switch (mode) {
    case "table":
      return `Extract the TABLE from this image.

Output TSV only (tabs between cells, newline between rows).
If header exists, include it as first row.
Preserve blank cells; for merged cells repeat value if clear, else leave blank.
Add 'Notes: merged cells present' only if applicable.`;

    case "chart":
      return `Analyze this CHART/GRAPH in detail.

${getChartInstructions()}`;

    case "diagram":
      return `Extract this DIAGRAM's structure.

Components: bullet list of main elements
Connections: A -> B (label) for each relationship
Labels: extract all text exactly`;

    case "text":
      return `Extract all visible TEXT from this image.

Extract in reading order. Preserve formatting, line breaks, and hierarchy.
For UI elements: [Button: "label"] or [Menu: item1, item2]
For code: preserve indentation exactly.
Just extract the text accurately - no analysis needed.`;

    case "general":
      return getUnifiedPrompt();
  }
}

function getChartInstructions(): string {
  return `Report in this EXACT format:

TYPE: [scatter/line/bar/pie/area/histogram/etc]
TITLE: [title or "none"]
X-AXIS: [label] range [min to max]
Y-AXIS: [label] range [min to max]
LEGEND: [color1]=name1, [color2]=name2, ...

SERIES ANALYSIS (for EACH visible series/color):
- [Series name/color]: center=(~x,~y), spread=[tight/wide/elongated], x=[min,max], y=[min,max]
  Pattern: [describe using these terms as applicable:]
    • linear increasing/decreasing (slope: gradual/steep/~45°)
    • exponential growth/decay
    • logarithmic curve
    • constant/flat/horizontal
    • vertical band/spike
    • parabolic/curved
    • scattered/no clear pattern
  Notable: [include any of these if present:]
    • steep drop-off at x≈[value]
    • sharp increase at x≈[value]
    • plateau between x≈[start] and x≈[end]
    • inflection point at (~x,~y)
    • outliers at (~x,~y)
    • asymptotic behavior approaching y≈[value]
    • peak/maximum at (~x,~y)
    • trough/minimum at (~x,~y)
    • discontinuity/gap at x≈[value]

CROSS-SERIES PATTERNS:
- Correlations: [positive/negative/none between which series]
- Clusters: [where multiple series overlap or group]
- Separations: [clear boundaries between series]
- Crossovers: [where series intersect, with coordinates]

Describe EVERY series with specific coordinates and pattern characteristics.`;
}

function getUnifiedPrompt(): string {
  return `Analyze this image and extract its content. First, silently identify the content type, then apply the appropriate extraction strategy.

═══ IF TEXT/UI/SCREENSHOT (documents, code, app UI, messages, articles) ═══
Extract all visible text in reading order.
Preserve formatting, line breaks, and hierarchy.
For UI: [Button: "label"], [Menu: item1, item2]
For code: preserve indentation exactly.
Keep it simple - just accurate text extraction.

═══ IF TABLE (structured rows and columns of data) ═══
Output TSV format only (tabs between cells, newlines between rows).
Include header row if present.
Preserve blank cells; note merged cells if present.

═══ IF CHART/GRAPH (data visualization with axes, series, trends) ═══
${getChartInstructions()}

═══ IF DIAGRAM (flowcharts, architecture, process flows, org charts) ═══
Components: bullet list of main elements
Connections: A -> B (label) for each relationship
Labels: extract all text exactly

═══ IF PHOTO/OTHER (photographs, artwork, mixed content) ═══
Brief factual description of key visible elements.
Extract any visible text.

IMPORTANT: Match your response complexity to the content. Simple text screenshots need simple extraction. Complex charts need detailed analysis with coordinates. Do not over-analyze simple content or under-analyze data visualizations.`;
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
