import { useState } from "react";
import { Card } from "./ui/card";
import { User } from "lucide-react";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { PatientModule } from "@/types/module";

function ModuleControl({
  module,
  onStart,
  isSessionActive,
}: {
  module: PatientModule;
  onStart: () => Promise<void>;
  isSessionActive: boolean;
}) {
  const [isActivating, setIsActivating] = useState<boolean>(false);

  const handleStartModule = async () => {
    if (isActivating) return;
    setIsActivating(true);
    try {
      await onStart();
    } catch (error) {
      console.error("Failed to start module:", error);
    } finally {
      setIsActivating(false);
    }
  };

  return (
    <Card className="w-full bg-white shadow-sm">
      <div className="p-6">
        <div className="flex items-start space-x-4">
          <div className="h-12 w-12 rounded-full bg-blue-100 flex items-center justify-center">
            <User className="h-6 w-6 text-blue-600" />
          </div>
          <div className="flex-1">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-medium text-gray-900">
                  {module.name}
                </h3>
                <p className="text-sm text-gray-500 mt-1">
                  {module.description}
                </p>
              </div>
              <Badge
                variant="secondary"
                className={
                  module.status === "active"
                    ? "bg-green-100 text-green-800"
                    : module.status === "completed"
                    ? "bg-blue-100 text-blue-800"
                    : "bg-gray-100 text-gray-800"
                }
              >
                {module.status.charAt(0).toUpperCase() + module.status.slice(1)}
              </Badge>
            </div>

            <div className="mt-4 flex items-center justify-between">
              <div className="text-sm text-gray-500">
                {module.lastAccessed && (
                  <span>
                    Last accessed:{" "}
                    {new Date(module.lastAccessed).toLocaleDateString()}
                  </span>
                )}
              </div>
              <Button
                onClick={handleStartModule}
                disabled={isActivating || isSessionActive}
                className={`${
                  isActivating || isSessionActive
                    ? "bg-gray-100 text-gray-600"
                    : "bg-blue-600 hover:bg-blue-700 text-white"
                }`}
              >
                {isActivating ? "Starting..." : "Start Session"}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}

export default ModuleControl;
