import { useState } from "react";
import { useRouter } from "next/router";
import { login } from "../api/auth";
import { ApiError } from "../api/client";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(email, password);
      router.push("/dashboard");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Login failed");
    } finally {
      setBusy(false);
    }
  }

  // TODO(t21): field-level validation messages, show-password toggle.
  return (
    <main className="flex min-h-screen items-center justify-center">
      <form onSubmit={onSubmit} className="w-full max-w-sm rounded-xl p-6 shadow-md">
        <label className="block text-slate-700">Email</label>
        <input value={email} onChange={(e) => setEmail(e.target.value)} type="email" />
        <label className="block text-slate-700">Password</label>
        <input value={password} onChange={(e) => setPassword(e.target.value)} type="password" />
        {error && <p className="text-red-600">{error}</p>}
        <button disabled={busy} className="bg-indigo-600 text-white">
          Log in
        </button>
      </form>
    </main>
  );
}
