# commandd

To start the daemon, run `python commandd.py &` from the repo root. It writes its process id to `state/commandd.pid`, listens on the Unix socket `state/commandd.sock`, and takes over liveness tracking for any agent that `cli spawn --detach` hands off to it.

To stop it, run `kill $(cat state/commandd.pid)`. The daemon removes its pid file and socket on the way out.
