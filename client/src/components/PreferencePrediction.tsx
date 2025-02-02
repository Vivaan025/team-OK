import React, { useState } from 'react';
import { 
  Card, 
  CardContent, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  MapPin, 
  Store, 
  Star, 
  ShoppingBag 
} from 'lucide-react';

interface StorePreferencePredictionProps {
  customerId: string;
}

interface RecommendedStore {
  name: string;
  address: string;
  city: string;
  pincode: string;
  ratings: {
    overall: number;
    service: number;
  };
  amenities: string[];
}

interface PreferencePrediction {
  predicted_category: string;
  confidence: number;
  recommended_store: RecommendedStore;
}

export const StorePreferencePrediction: React.FC<StorePreferencePredictionProps> = ({ customerId }) => {
  const [prediction, setPrediction] = useState<PreferencePrediction | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchPreferencePrediction = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await fetch(`http://localhost:8000/api/predict-preference/${customerId}?days=500`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch preference prediction');
      }
      
      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
      setPrediction(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex justify-between items-center">
          <div className="flex items-center">
            <ShoppingBag className="mr-2" />
            Store Preference Prediction
          </div>
          <Button 
            variant="outline" 
            size="sm"
            onClick={fetchPreferencePrediction}
            disabled={isLoading}
          >
            {isLoading ? 'Predicting...' : 'Predict Next Store'}
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {error && (
          <div className="text-red-500 mb-4">
            {error}
          </div>
        )}
        
        {prediction ? (
          <div className="grid gap-4">
            <div>
              <p className="text-muted-foreground">Predicted Category</p>
              <Badge variant="secondary">{prediction.predicted_category}</Badge>
            </div>
            
            <div>
              <p className="text-muted-foreground">Confidence</p>
              <p className="font-bold">
                {(prediction.confidence * 100).toFixed(2)}%
              </p>
            </div>
            
            <Card className="border-dashed">
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <div className="flex items-center">
                    <Store className="mr-2" />
                    Recommended Store
                  </div>
                  <div className="flex items-center">
                    <Star className="mr-1 text-yellow-500" />
                    {prediction.recommended_store.ratings.overall}/5
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-muted-foreground">Name</p>
                    <p className="font-bold">{prediction.recommended_store.name}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Address</p>
                    <p className="font-bold">
                      {prediction.recommended_store.address}
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Pincode</p>
                    <p className="font-bold">{prediction.recommended_store.pincode}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Amenities</p>
                    <div className="flex flex-wrap gap-2">
                      {prediction.recommended_store.amenities.map((amenity) => (
                        <Badge key={amenity} variant="outline">{amenity}</Badge>
                      ))}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        ) : (
          <p className="text-muted-foreground text-center">
            Click 'Predict Next Store' to get personalized recommendations
          </p>
        )}
      </CardContent>
    </Card>
  );
};