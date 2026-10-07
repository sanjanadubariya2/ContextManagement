import { apiFetch } from "./client";

export interface User {
  id: string;
  email: string;
}

export async function login(email: string, password: string): Promise<void> {
  // TODO(t21): call POST /api/auth/login; the server sets the cookie.
  await apiFetch<void>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export async function logout(): Promise<void> {
  await apiFetch<void>("/api/auth/logout", { method: "POST" });
}

export async function currentUser(): Promise<User | null> {
  try {
    return await apiFetch<User>("/api/auth/me");
  } catch {
    return null;
  }
}
