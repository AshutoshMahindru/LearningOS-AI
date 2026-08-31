export function jsonResponse(data: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json', ...headers },
  });
}

export function errorEnvelope(
  code: string,
  message: string,
  details: Record<string, unknown> = {},
  status = 400,
): Response {
  return jsonResponse({ error: { code, message, details } }, status);
}
