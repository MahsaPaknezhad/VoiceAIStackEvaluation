import { ThemeProvider } from "@pipecat-ai/voice-ui-kit";
import { createRoot } from "react-dom/client";
import { NABVoiceCoach } from "./components/NABVoiceCoach";
import FullFlowTestPage from "./components/FullFlowTestPage";

//@ts-ignore - fontsource-variable/geist is not typed
import "@fontsource-variable/geist";
//@ts-ignore - fontsource-variable/geist is not typed
import "@fontsource-variable/geist-mono";

// Toggle between full flow test and main app
const showFullFlowTest = false;

createRoot(document.getElementById("root")!).render(
  <ThemeProvider>
    <div className="min-h-screen bg-background">
      {showFullFlowTest ? (
        <FullFlowTestPage />
      ) : (
        <NABVoiceCoach
          connectParams={{
            webrtcRequestParams: {
              endpoint: "/api/offer",
            },
          }}
        />
      )}
    </div>
  </ThemeProvider>
);