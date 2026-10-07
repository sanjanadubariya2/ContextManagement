import { useEffect, useState } from "react";
import { currentUser, type User } from "../api/auth";

export function useSession() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    currentUser()
      .then(setUser)
      .finally(() => setLoading(false));
  }, []);

  return { user, loading, setUser };
}
