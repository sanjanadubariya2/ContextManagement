# Starter site requirements

## Authentication
- Users log in with email and password on /login.
- An invalid email format is rejected with 400; wrong credentials with 401.
- After login the user lands on /dashboard.
- Sessions last 30 minutes of inactivity; a silent refresh keeps active users logged in.
- Logging out clears the session everywhere in the browser.

## Dashboard
- The dashboard shows project and open-task counters and the five most recent activity items.
- Unauthenticated visitors to /dashboard are redirected to /login.

## API integration
- The dashboard shows the weather for the user's city from an external weather API.
- External calls go through the backend, are cached for 10 minutes, and fail soft:
  the dashboard still renders when the weather API is down.

## Quality
- Every API endpoint has a backend test; the login flow has an end-to-end test.
- Errors use application/problem+json.
