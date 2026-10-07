import { logout } from "../api/auth";
import { useSession } from "../hooks/useSession";

export default function Header() {
  const { user, setUser } = useSession();
  if (!user) return null;
  return (
    <header className="flex justify-between p-4 text-slate-700">
      <span>{user.email}</span>
      <button
        onClick={async () => {
          await logout();
          setUser(null);
        }}
      >
        Log out
      </button>
    </header>
  );
}
