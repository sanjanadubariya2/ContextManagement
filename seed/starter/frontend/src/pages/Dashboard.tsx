import { useEffect, useState } from "react";
import { fetchSummary, fetchWeather, type Summary, type Weather } from "../api/dashboard";
import { useSession } from "../hooks/useSession";
import Header from "../components/Header";

export default function DashboardPage() {
  const { user, loading } = useSession();
  const [summary, setSummary] = useState<Summary | null>(null);
  const [weather, setWeather] = useState<Weather | null>(null);

  useEffect(() => {
    if (!user) return;
    fetchSummary().then(setSummary);
    fetchWeather("Bengaluru").then(setWeather);
  }, [user]);

  if (loading) return <p>Loading…</p>;
  // TODO(t30): redirect unauthenticated visitors to /login.
  return (
    <>
      <Header />
      <section className="grid grid-cols-2 gap-4 p-6">
        <div>Projects: {summary?.projects ?? "–"}</div>
        <div>Open tasks: {summary?.open_tasks ?? "–"}</div>
        {weather && (
          <div>
            {weather.city}: {weather.temp_c}°C, {weather.condition}
          </div>
        )}
      </section>
    </>
  );
}
