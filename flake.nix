{
  description = "Wildcams Python development environment with uv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Python and uv
            python3
            uv
            
            # Common development tools
            git
            
            # System libraries that uv packages need
            stdenv.cc.cc.lib
            zlib
            
            # OpenGL libraries for OpenCV
            libGL
            libGLU
            xorg.libX11
            xorg.libXext
            
            # Additional libraries for OpenCV
            glib
            
            # Custom scripts for daemon management
            (pkgs.writeShellScriptBin "start" ''
              PID_FILE="$PWD/.sd_watcher.pid"
              LOG_FILE="$PWD/.sd_watcher.log"
              
              # Stop existing daemon if running
              if [[ -f "$PID_FILE" ]]; then
                pid=$(cat "$PID_FILE")
                if kill -0 "$pid" 2>/dev/null; then
                  echo "🔄 Stopping existing daemon (PID: $pid)..."
                  kill -TERM "$pid"
                  
                  # Wait for graceful shutdown
                  count=0
                  while kill -0 "$pid" 2>/dev/null && [[ $count -lt 5 ]]; do
                    sleep 1
                    ((count++))
                  done
                  
                  # Force kill if still running
                  if kill -0 "$pid" 2>/dev/null; then
                    kill -9 "$pid" 2>/dev/null
                  fi
                  echo "✅ Previous daemon stopped"
                fi
                # Clean up PID file
                rm -f "$PID_FILE"
              fi
              
              echo "🎥 Starting SD card watcher daemon..."
              "$PWD/sd_watcher.py" --daemon --pid-file "$PID_FILE" --log-file "$LOG_FILE" &
              start_pid=$!
              
              # Wait briefly for daemon to start and write PID file
              sleep 0.5
              wait $start_pid 2>/dev/null
              
              if [[ -f "$PID_FILE" ]]; then
                pid=$(cat "$PID_FILE")
                echo "✅ SD card watcher started (PID: $pid)"
                echo "📋 View logs: logs"
                echo "🛑 Stop with: stop"
              else
                echo "❌ Failed to start daemon"
                exit 1
              fi
            '')
            
            (pkgs.writeShellScriptBin "stop" ''
              PID_FILE="$PWD/.sd_watcher.pid"
              if [[ -f "$PID_FILE" ]]; then
                pid=$(cat "$PID_FILE")
                if kill -0 "$pid" 2>/dev/null; then
                  echo "🛑 Stopping SD card watcher (PID: $pid)..."
                  kill -TERM "$pid"
                  
                  # Wait for graceful shutdown
                  count=0
                  while kill -0 "$pid" 2>/dev/null && [[ $count -lt 5 ]]; do
                    sleep 1
                    ((count++))
                  done
                  
                  # Force kill if still running
                  if kill -0 "$pid" 2>/dev/null; then
                    kill -9 "$pid" 2>/dev/null
                  fi
                  echo "✅ SD card watcher stopped"
                else
                  echo "⚠️  Process not running"
                fi
                rm -f "$PID_FILE"
              else
                echo "⚠️  No PID file found - daemon not running"
              fi
            '')
            
            (pkgs.writeShellScriptBin "logs" ''
              LOG_FILE="$PWD/.sd_watcher.log"
              if [[ -f "$LOG_FILE" ]]; then
                echo "📋 Following SD card watcher logs (Ctrl+C to exit)..."
                tail -f "$LOG_FILE"
              else
                echo "⚠️  No log file found at $LOG_FILE"
              fi
            '')
            
            (pkgs.writeShellScriptBin "check" ''
              PID_FILE="$PWD/.sd_watcher.pid"
              if [[ -f "$PID_FILE" ]]; then
                pid=$(cat "$PID_FILE")
                if kill -0 "$pid" 2>/dev/null; then
                  echo "✅ SD card watcher running (PID: $pid)"
                  echo "📋 Log file: $PWD/.sd_watcher.log"
                  echo "🛑 Stop with: stop"
                  echo "📋 View logs: logs"
                else
                  echo "❌ PID file exists but process not running"
                  rm -f "$PID_FILE"
                fi
              else
                echo "⚠️  SD card watcher not running"
              fi
            '')
            
            (pkgs.writeShellScriptBin "process-ff" ''
              echo "🎬 Starting wildlife video processing (Full Frame)..."
              echo "📊 Mode: Full frame ML ensemble processing"
              "$PWD/process_fullframe.py" "$@"
            '')
            
            (pkgs.writeShellScriptBin "process-md" ''
              echo "🎬 Starting wildlife video processing (Motion Detection)..."
              echo "📊 Mode: Motion detection + crop-based ML processing"
              "$PWD/process_motiondetection.py" "$@"
            '')
            
            (pkgs.writeShellScriptBin "process" ''
              "$PWD/process.py" "$@"
            '')
          ];

          shellHook = ''
            # Set library path for uv packages
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.libGL}/lib:${pkgs.libGLU}/lib:${pkgs.xorg.libX11}/lib:${pkgs.xorg.libXext}/lib:${pkgs.glib.out}/lib:$LD_LIBRARY_PATH"
            
            echo "🐍 Wildcams development environment loaded!"
            echo "📦 uv version: $(uv --version)"
            echo "🐍 Python version: $(python --version)"
            echo ""
            echo "📱 SD Card Watcher Commands:"
            echo "  start - Start the daemon"
            echo "  stop  - Stop the daemon" 
            echo "  logs  - Follow log output"
            echo "  check - Check daemon status"
            echo ""
            echo "🎬 Video Processing Commands:"
            echo "  process    - Unified processor with strategy selection"
            echo "  process-ff - Full frame ML ensemble processing"
            echo "  process-md - Motion detection + crop processing"
            echo ""
            echo "💡 Usage examples:"
            echo "  process -s ff -v 7 8 9        # Full frame strategy"
            echo "  process -s md -v 7 8 9        # Motion detection strategy"
            echo "  process-ff --videos 7 8 9     # Direct full frame processing"
            echo "  process-md --videos IMG_0015.MP4  # Direct motion detection processing"
          '';
        };
      });
}