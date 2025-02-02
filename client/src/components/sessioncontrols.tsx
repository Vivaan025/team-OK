'use client';

import { useState } from 'react';
import { Play, Loader } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface ModuleControlProps {
  moduleName: string;
  moduleDescription?: string;
  moduleStatus?: 'active' | 'inactive' | 'completed';
  startSession: () => Promise<void>;
  isSessionActive: boolean;
}

export default function ModuleControl({
  moduleName,
  moduleDescription = 'Start patient consultation module',
  moduleStatus = 'inactive',
  startSession,
  isSessionActive
}: ModuleControlProps) {
  const [isActivating, setIsActivating] = useState<boolean>(false);

  const handleStartModule = async () => {
    if (isActivating) return;
    setIsActivating(true);
    try {
      await startSession();
    } catch (error) {
      console.error('Failed to start module:', error);
    } finally {
      setIsActivating(false);
    }
  };

  const getStatusBadge = () => {
    const statusConfig = {
      active: { class: 'bg-green-100 text-green-800', text: 'Active' },
      inactive: { class: 'bg-gray-100 text-gray-800', text: 'Ready to Start' },
      completed: { class: 'bg-blue-100 text-blue-800', text: 'Completed' }
    };

    const config = statusConfig[moduleStatus];
    return (
      <Badge variant="secondary" className={config.class}>
        {config.text}
      </Badge>
    );
  };

  return (
    <Card className="w-full max-w-sm p-4 bg-white shadow-sm hover:shadow-md transition-shadow duration-200">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-medium text-gray-900">{moduleName}</h3>
          <p className="text-sm text-gray-500 mt-1">{moduleDescription}</p>
        </div>
        {getStatusBadge()}
      </div>

      <Button
        onClick={handleStartModule}
        disabled={isActivating || isSessionActive}
        className={`w-full ${
          isActivating || isSessionActive
            ? 'bg-gray-100 text-gray-600'
            : 'bg-blue-600 hover:bg-blue-700 text-white'
        }`}
      >
        {isActivating ? (
          <div className="flex items-center justify-center space-x-2">
            <Loader className="h-4 w-4 animate-spin" />
            <span>Starting...</span>
          </div>
        ) : (
          <div className="flex items-center justify-center space-x-2">
            <Play className="h-4 w-4" />
            <span>Start Module</span>
          </div>
        )}
      </Button>
    </Card>
  );
}