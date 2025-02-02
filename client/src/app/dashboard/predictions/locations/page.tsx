"use client"
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
  Card, 
  CardContent, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card";
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { 
  MapPin, 
  ChevronUp, 
  ChevronDown, 
  RefreshCw,
  Maximize2,
  Minimize2
} from 'lucide-react';

interface StoreLocation {
  latitude: number;
  longitude: number;
  estimated_revenue: number;
  nearby_cities: string[];
}

declare global {
  interface Window {
    L: any;
  }
}

const StorePlacementRecommendations: React.FC = () => {
  const [locations, setLocations] = useState<StoreLocation[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortOrder, setSortOrder] = useState<'desc' | 'asc'>('desc');
  const [selectedLocation, setSelectedLocation] = useState<StoreLocation | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isMapReady, setIsMapReady] = useState(false);
  const mapInstanceRef = useRef<any>(null);
  const mapRef = useRef<HTMLDivElement>(null);
  const markersRef = useRef<any[]>([]);

  // Load Leaflet resources
  useEffect(() => {
    const loadLeaflet = async () => {
      try {
        // Add CSS
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css';
        document.head.appendChild(link);

        // Add custom styles
        const style = document.createElement('style');
        style.textContent = `
          .leaflet-container {
            height: 100%;
            width: 100%;
            z-index: 0 !important;
          }
          .leaflet-popup-content {
            font-size: 14px;
            line-height: 1.5;
          }
          .location-popup .revenue {
            color: #16a34a;
            font-weight: 600;
          }
        `;
        document.head.appendChild(style);

        // Load Leaflet script
        await new Promise<void>((resolve, reject) => {
          const script = document.createElement('script');
          script.src = 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js';
          script.async = true;
          script.onload = () => resolve();
          script.onerror = () => reject(new Error('Failed to load Leaflet'));
          document.body.appendChild(script);
        });

        setIsMapReady(true);
      } catch (err) {
        setError('Failed to load map resources');
        console.error('Error loading Leaflet:', err);
      }
    };

    loadLeaflet();

    // Cleanup function to remove added elements
    return () => {
      const leafletLink = document.querySelector('link[href*="leaflet.css"]');
      const leafletScript = document.querySelector('script[src*="leaflet.js"]');
      if (leafletLink) document.head.removeChild(leafletLink);
      if (leafletScript) document.body.removeChild(leafletScript);
    };
  }, []);

  // Initialize map
  useEffect(() => {
    if (!isMapReady || !mapRef.current || mapInstanceRef.current) return;

    try {
      const mapInstance = window.L.map(mapRef.current).setView([20.5937, 78.9629], 4);
      
      window.L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
      }).addTo(mapInstance);

      mapInstanceRef.current = mapInstance;
    } catch (err) {
      setError('Failed to initialize map');
      console.error('Error initializing map:', err);
    }

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, [isMapReady]);

  // Fetch store locations
  const fetchStoreLocations = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await fetch('http://localhost:8000/api/store-placement-recommendations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ num_recommendations: 7 })
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch store placement recommendations');
      }
      
      const data = await response.json();
      setLocations(data);
      
      if (data.length > 0) {
        setSelectedLocation(data[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  // Update markers when locations or selected location changes
  useEffect(() => {
    if (!mapInstanceRef.current || !window.L || locations.length === 0) return;

    // Clear existing markers
    markersRef.current.forEach(marker => marker.remove());
    markersRef.current = [];

    // Add new markers
    locations.forEach(location => {
      const marker = window.L.marker([location.latitude, location.longitude])
        .bindPopup(`
          <div class="location-popup">
            <strong>Location Details</strong><br>
            Coordinates: ${location.latitude.toFixed(4)}, ${location.longitude.toFixed(4)}<br>
            Estimated Revenue: ₹${location.estimated_revenue.toLocaleString()}<br>
          </div>
        `);

      marker.addTo(mapInstanceRef.current);
      markersRef.current.push(marker);

      if (selectedLocation && 
          selectedLocation.latitude === location.latitude && 
          selectedLocation.longitude === location.longitude) {
        marker.openPopup();
        mapInstanceRef.current.setView([location.latitude, location.longitude], 7);
      }
    });
  }, [locations, selectedLocation]);

  // Initial data fetch
  useEffect(() => {
    fetchStoreLocations();
  }, []);

  // Sort locations by estimated revenue
  const sortedLocations = React.useMemo(() => {
    return [...locations].sort((a, b) => 
      sortOrder === 'desc' 
        ? b.estimated_revenue - a.estimated_revenue 
        : a.estimated_revenue - b.estimated_revenue
    );
  }, [locations, sortOrder]);

  // Toggle sort order
  const toggleSortOrder = () => {
    setSortOrder(prev => prev === 'desc' ? 'asc' : 'desc');
  };

  // Handle location selection
  const handleLocationSelect = useCallback((location: StoreLocation) => {
    setSelectedLocation(location);
    if (mapInstanceRef.current) {
      mapInstanceRef.current.setView([location.latitude, location.longitude], 7);
      markersRef.current.forEach(marker => {
        const markerLatLng = marker.getLatLng();
        if (markerLatLng.lat === location.latitude && 
            markerLatLng.lng === location.longitude) {
          marker.openPopup();
        }
      });
    }
  }, []);

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="flex items-center space-x-2">
          <RefreshCw className="animate-spin" />
          <p>Loading store placement recommendations...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col justify-center items-center h-screen">
        <div className="text-red-500 mb-4">{error}</div>
        <Button onClick={fetchStoreLocations}>
          <RefreshCw className="mr-2" size={16} /> Try Again
        </Button>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold flex items-center">
          <MapPin className="mr-2" /> Store Placement Recommendations
        </h1>
        <div className="flex items-center space-x-2">
          <Button 
            variant="outline" 
            size="sm"
            onClick={fetchStoreLocations}
          >
            <RefreshCw className="mr-2" size={16} /> Refresh
          </Button>
          <Button 
            variant="outline" 
            size="sm"
            onClick={() => setIsFullscreen(!isFullscreen)}
          >
            {isFullscreen ? (
              <Minimize2 className="mr-2" size={16} />
            ) : (
              <Maximize2 className="mr-2" size={16} />
            )}
            {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
          </Button>
        </div>
      </div>

      <div className={`grid grid-cols-1 ${isFullscreen ? 'lg:grid-cols-1' : 'lg:grid-cols-2'} gap-6`}>
        {/* Location Table */}
        <Card>
          <CardHeader>
            <CardTitle className="flex justify-between items-center">
              Location Details
              <Button 
                variant="ghost" 
                size="sm"
                onClick={toggleSortOrder}
              >
                {sortOrder === 'desc' ? (
                  <ChevronDown className="mr-2" size={16} />
                ) : (
                  <ChevronUp className="mr-2" size={16} />
                )}
                Sort by Revenue
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Location</TableHead>
                  <TableHead>Estimated Revenue</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {sortedLocations.map((location, index) => (
                  <TableRow 
                    key={index}
                    onClick={() => handleLocationSelect(location)}
                    className={`cursor-pointer hover:bg-muted ${
                      selectedLocation?.latitude === location.latitude && 
                      selectedLocation?.longitude === location.longitude ? 'bg-muted' : ''
                    }`}
                  >
                    <TableCell>
                      <div className="flex items-center">
                        <MapPin className="mr-2 text-muted-foreground" size={16} />
                        {location.latitude.toFixed(4)}, {location.longitude.toFixed(4)}
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="secondary">
                        ₹{location.estimated_revenue.toLocaleString()}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        {/* Map Card */}
        <Card className={`${isFullscreen ? 'fixed inset-0 z-50 m-0' : ''}`}>
          <CardHeader>
            <CardTitle>Store Placement Map</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <div 
              ref={mapRef}
              className="rounded-lg"
              style={{ height: isFullscreen ? 'calc(100vh - 140px)' : '400px', width: '100%' }}
            ></div>
          </CardContent>
        </Card>
      </div>

      {/* Selected Location Details */}
      {selectedLocation && !isFullscreen && (
        <Card className="mt-6">
          <CardHeader>
            <CardTitle>Selected Location Details</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <p className="text-muted-foreground">Coordinates</p>
                <p className="font-bold">
                  Latitude: {selectedLocation.latitude.toFixed(4)}, 
                  Longitude: {selectedLocation.longitude.toFixed(4)}
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">Estimated Revenue</p>
                <Badge variant="secondary" className="text-lg">
                  ₹{selectedLocation.estimated_revenue.toLocaleString()}
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default StorePlacementRecommendations;