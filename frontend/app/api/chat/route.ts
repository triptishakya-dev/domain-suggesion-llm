const modelApiUrl = process.env.MODEL_API_URL ?? "http://127.0.0.1:8000";

export async function POST(request: Request) {
  let payload: { message?: unknown };
  try { payload = await request.json(); } catch { return Response.json({ error: "Request body must be valid JSON." }, { status: 400 }); }
  const message = typeof payload.message === "string" ? payload.message.trim() : "";
  if (!message) return Response.json({ error: "Please enter a message." }, { status: 400 });
  if (message.length > 1000) return Response.json({ error: "Messages are limited to 1,000 characters." }, { status: 400 });
  try {
    const response = await fetch(`${modelApiUrl}/api/suggest`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message }), cache: "no-store", signal: AbortSignal.timeout(60_000) });
    const body: unknown = await response.json();
    if (!response.ok) {
      const detail = typeof body === "object" && body !== null && "detail" in body ? String(body.detail) : "The model could not process that message.";
      return Response.json({ error: detail }, { status: response.status });
    }
    return Response.json(body);
  } catch { return Response.json({ error: "The model service is unavailable. Start the Python API and try again." }, { status: 503 }); }
}
