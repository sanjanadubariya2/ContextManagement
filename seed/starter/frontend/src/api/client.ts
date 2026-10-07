// Thin fetch wrapper. Auth rides on the HttpOnly cookie, so every request
// sends credentials and nothing here ever touches a token directly.

export class ApiError extends Error {
  constructor(
    public status: number,
    public problem: { title?: string; detail?: string } = {},
  ) {
    super(problem.title ?? `HTTP ${status}`);
  }
}

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(path, {
    ...init,
    credentials: "include",
    headers: { "Content-Type": "application/json", ...(init.headers ?? {}) },
  });
  if (!res.ok) {
    const problem = await res.json().catch(() => ({}));
    throw new ApiError(res.status, problem);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}
