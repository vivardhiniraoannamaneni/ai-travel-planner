import express, { Request, Response } from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

// Middleware for parsing JSON requests
app.use(express.json());

// Initialize Gemini SDK securely
const apiKey = process.env.GEMINI_API_KEY;
const ai = new GoogleGenAI({
  apiKey: apiKey,
  httpOptions: {
    headers: {
      "User-Agent": "aistudio-build",
    },
  },
});

// JSON Schema for structured itinerary generator
const ItinerarySchema = {
  type: Type.OBJECT,
  properties: {
    tripOverview: {
      type: Type.OBJECT,
      properties: {
        title: { type: Type.STRING, description: "Eye-catching title for the trip" },
        description: { type: Type.STRING, description: "Captivating description summarizing the entire trip experience based on interests" }
      },
      required: ["title", "description"]
    },
    dayByDayPlan: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          day: { type: Type.INTEGER, description: "The day number" },
          theme: { type: Type.STRING, description: "Daily theme, e.g. Exploring Historic Centers" },
          morning: { type: Type.STRING, description: "Morning activities: locations, transport tips, sights, breakfast options" },
          afternoon: { type: Type.STRING, description: "Afternoon activities: lunch recommendation, main exploration target, tour" },
          evening: { type: Type.STRING, description: "Evening activities: dinner choice, nightlife or relaxation activity, beautiful sunset view spotting" }
        },
        required: ["day", "theme", "morning", "afternoon", "evening"]
      }
    },
    estimatedDailyBudget: {
      type: Type.OBJECT,
      properties: {
        accommodation: { type: Type.STRING, description: "Estimated hotel/stay cost per day" },
        food: { type: Type.STRING, description: "Estimated dining/snacks cost per day" },
        activities: { type: Type.STRING, description: "Estimated sights/tours cost per day" },
        transport: { type: Type.STRING, description: "Estimated local transportation cost per day" },
        total: { type: Type.STRING, description: "Calculated range or total estimated daily budget" }
      },
      required: ["accommodation", "food", "activities", "transport", "total"]
    },
    recommendedFoods: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          name: { type: Type.STRING, description: "Food name plus description, e.g. Tacos al Pastor" },
          description: { type: Type.STRING, description: "Why try it and where to find the best version" }
        },
        required: ["name", "description"]
      }
    },
    travelTips: {
      type: Type.ARRAY,
      items: { type: Type.STRING, description: "Practical local tip (safety, transport, culture, packing)" }
    },
    bestPlacesToVisit: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          name: { type: Type.STRING, description: "Attraction name" },
          description: { type: Type.STRING, description: "Key highlight and why it's a must-visit" }
        },
        required: ["name", "description"]
      }
    }
  },
  required: [
    "tripOverview",
    "dayByDayPlan",
    "estimatedDailyBudget",
    "recommendedFoods",
    "travelTips",
    "bestPlacesToVisit"
  ]
};

// API Endpoint for generating itinerary
app.post("/api/generate-itinerary", async (req: Request, res: Response): Promise<void> => {
  try {
    const { destination, duration, budget, interests } = req.body;

    if (!destination || !duration || !budget) {
      res.status(400).json({ error: "Missing required fields: destination, duration, and budget are required." });
      return;
    }

    const durationNum = parseInt(duration, 10);
    const interestsStr = (interests && Array.isArray(interests) && interests.length > 0)
      ? interests.join(", ")
      : "General Travel";

    if (isNaN(durationNum) || durationNum <= 0) {
      res.status(400).json({ error: "Duration must be a positive number of days." });
      return;
    }

    const systemInstruction = 
      "You are a local custom concierge and travel expert. Create a highly descriptive and structured travel itinerary. " +
      "Tailor details specifically to user interests and budget level (Budget, Moderate, Luxury). " +
      "Ensure recommended foods and places are ultra-authentic and highly rated.";

    const userPrompt = `Create a detailed travel itinerary.

Destination: ${destination}
Duration: ${durationNum} days
Budget: ${budget}
Interests: ${interestsStr}

Generate:
- Trip Overview
- Day-by-Day Plan
- Morning Activities
- Afternoon Activities
- Evening Activities
- Estimated Daily Budget
- Recommended Foods
- Travel Tips
- Best Places To Visit

Keep the response well formatted and easy to read.`;

    // Check if API key is set
    if (!process.env.GEMINI_API_KEY) {
      res.status(500).json({
        error: "GEMINI_API_KEY environment variable is not set. Please set it in Settings > Secrets."
      });
      return;
    }

    // High availability caller with exponential backoff and fallback model
    const executeCallWithBackup = async () => {
      const modelsChain = ["gemini-3.5-flash", "gemini-3.1-flash-lite"];
      let lastError: any = null;

      for (const modelName of modelsChain) {
        const maxRetries = 2; // 3 attempts total per model
        for (let attempt = 0; attempt <= maxRetries; attempt++) {
          try {
            console.log(`Dispatching itinerary request to Gemini model "${modelName}" (Attempt ${attempt + 1}/${maxRetries + 1})...`);
            const response = await ai.models.generateContent({
              model: modelName,
              contents: userPrompt,
              config: {
                systemInstruction,
                responseMimeType: "application/json",
                responseSchema: ItinerarySchema,
                temperature: 0.75,
              },
            });
            return response;
          } catch (err: any) {
            lastError = err;
            const errMsg = err?.message || String(err);
            console.warn(`Attempt ${attempt + 1} with model "${modelName}" failed! Reason: ${errMsg}`);
            
            // If it's a structural or auth error, retrying won't help so rethrow immediately
            const statusCode = err?.status || err?.code;
            if (statusCode && statusCode >= 400 && statusCode < 500 && statusCode !== 429) {
              throw err;
            }

            // Perform active sleep if we are going to retry
            if (attempt < maxRetries) {
              const backoffMs = Math.pow(2, attempt) * 1200;
              console.log(`Waiting for ${backoffMs}ms before retrying...`);
              await new Promise((resolve) => setTimeout(resolve, backoffMs));
            }
          }
        }
        console.warn(`Fallback triggered: Model "${modelName}" was unsuccessful. Escalating to the next candidate...`);
      }
      throw lastError;
    };

    const response = await executeCallWithBackup();

    const responseText = response.text;
    if (!responseText) {
      throw new Error("No response text received from Gemini API.");
    }

    const parsedItinerary = JSON.parse(responseText.trim());
    res.json(parsedItinerary);

  } catch (err: any) {
    console.error("Error generating itinerary:", err);
    res.status(500).json({
      error: "Failed to generate travel itinerary. Please verify your input or try again.",
      details: err.message || String(err)
    });
  }
});

// Configure Vite or Serve static assets
async function setupServer() {
  if (process.env.NODE_ENV !== "production") {
    console.log("Starting server in DEVELOPMENT mode with Vite Middleware...");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    console.log("Starting server in PRODUCTION mode...");
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server listening on Port ${PORT}`);
  });
}

setupServer().catch((err) => {
  console.error("Failed to start server:", err);
});
