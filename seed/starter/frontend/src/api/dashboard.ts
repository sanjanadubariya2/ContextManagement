import { apiFetch } from "./client";

export interface Summary {
  projects: number;
  open_tasks: number;
  recent: string[];
}

export interface Weather {
  city: string;
  temp_c: number;
  condition: string;
}

export function fetchSummary(): Promise<Summary> {
  return apiFetch<Summary>("/api/dashboard/summary");
}

export async function fetchWeather(city: string): Promise<Weather | null> {
  // Fail soft: the dashboard renders without weather when the upstream is down.
  try {
    return await apiFetch<Weather>(`/api/integrations/weather?city=${encodeURIComponent(city)}`);
  } catch {
    return null;
  }
}
