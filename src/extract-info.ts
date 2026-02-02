import { Clipboard, Toast, showToast, getPreferenceValues, closeMainWindow, showHUD } from "@raycast/api";
import { execSync } from "child_process";
import fs from "fs";
import path from "path";
import os from "os";
import { OpenAI } from "openai";

type Mode = "table" | "chart" | "diagram" | "general";

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
  const userPrompt =
    "Classify this clipboard image into one pipeline: table, chart, diagram, or general. Choose the one that maximizes extraction accuracy. Return JSON with keys: type, format, priorities (array of strings).";

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
    "Start output directly with content. Do not begin with phrases like 'This image' or 'The image'.",
    "Extract ALL visible text exactly when readable; preserve capitalization and units.",
    "Never invent numbers. If unreadable, output [illegible] or [uncertain].",
    "No markdown code fences.",
    "Keep output compact but complete for the selected mode.",
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
Output in this order:
Title (if present)
Axes: X: <label> (<units if shown>), Y: <label> (<units if shown>)
Legend/series names
Data rows each as: <series>\\t<x>\\t<y> for readable points; prefix approximate values with ~.
Notes: list uncertainties/unreadable areas; do NOT guess points not readable.
`;
      break;
    case "diagram":
      modeInstructions = `
Output:
Components: bullet list of main objects/regions.
Labels: each as '<label text> -> <what it points to/describes>'.
Relationships: bullet list 'A -> B (label if present)' for arrows/flows.
Extract every label exactly.
`;
      break;
    case "general":
      modeInstructions = `
Output:
Short noun-phrase title/summary.
Visible text: lines of OCR-like text in reading order.
Key details: bullet list of important objects/regions with attributes.
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
