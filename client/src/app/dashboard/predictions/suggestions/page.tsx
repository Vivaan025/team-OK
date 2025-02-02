"use client"
import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { 
  Clock, 
  Package, 
  DollarSign, 
  Users,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  Store,
  Loader2,
  LucideIcon
} from 'lucide-react';

interface Location {
  city: string;
  state: string;
}

interface Ratings {
  overall: number;
}

interface StoreData {
  store_id: string;
  name: string;
  category: string;
  location: Location;
  ratings: Ratings;
}

interface SuggestionMetrics {
  [key: string]: number | number[] | string;
}

interface Suggestion {
  type: string;
  suggestion: string;
  priority: 'High' | 'Medium' | 'Low';
  metrics: SuggestionMetrics;
  impact_estimate: number;
  implementation_difficulty: string;
  timeframe: string;
}

interface SuggestionsByType {
  product_suggestions: Suggestion[];
  pricing_suggestions: Suggestion[];
  timing_suggestions: Suggestion[];
  customer_suggestions: Suggestion[];
}

interface ExpandedSections {
  [key: string]: boolean;
}

interface PriorityBadgeProps {
  priority: 'High' | 'Medium' | 'Low';
}

interface ImpactBadgeProps {
  value: number;
}

interface SuggestionCardProps {
  title: string;
  icon: LucideIcon;
  suggestions?: Suggestion[];
  type: string;
}

const StoreSuggestionsDashboard: React.FC = () => {
  const [stores, setStores] = useState<StoreData[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [selectedStore, setSelectedStore] = useState<StoreData | null>(null);
  const [suggestions, setSuggestions] = useState<SuggestionsByType | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expandedSections, setExpandedSections] = useState<ExpandedSections>({
    customer: true,
    product: true,
    pricing: true,
    timing: true
  });

  useEffect(() => {
    fetchStores();
  }, []);

  const fetchStores = async (): Promise<void> => {
    try {
      const response = await fetch('http://localhost:8000/api/stores');
      if (!response.ok) throw new Error('Failed to fetch stores');
      const data: StoreData[] = await response.json();
      setStores(data);
    } catch (err) {
      setError('Failed to load stores');
      console.error('Error:', err);
    }
  };

  const fetchSuggestions = async (storeId: string): Promise<void> => {
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/api/suggestions/${storeId}?days=90`);
      if (!response.ok) throw new Error('Failed to fetch suggestions');
      const data: SuggestionsByType = await response.json();
      setSuggestions(data);
      setSelectedStore(stores.find(s => s.store_id === storeId) || null);
    } catch (err) {
      setError('Failed to load suggestions');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const toggleSection = (id: string): void => {
    setExpandedSections(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  const PriorityBadge: React.FC<PriorityBadgeProps> = ({ priority }) => {
    const colors = {
      High: 'bg-red-100 text-red-800',
      Medium: 'bg-yellow-100 text-yellow-800',
      Low: 'bg-green-100 text-green-800'
    };
    
    return (
      <span className={`px-2 py-1 rounded-full text-xs font-semibold ${colors[priority]}`}>
        {priority}
      </span>
    );
  };

  const ImpactBadge: React.FC<ImpactBadgeProps> = ({ value }) => {
    const formattedValue = new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0
    }).format(value);

    return (
      <div className="flex items-center gap-1 text-emerald-600 font-medium">
        <DollarSign className="w-4 h-4" />
        {formattedValue}
      </div>
    );
  };

  const SuggestionCard: React.FC<SuggestionCardProps> = ({ title, icon: Icon, suggestions = [], type }) => {
    const isExpanded = expandedSections[type];

    if (!suggestions?.length) return null;

    return (
      <Card className="mb-6">
        <CardHeader 
          className="cursor-pointer hover:bg-gray-50"
          onClick={() => toggleSection(type)}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Icon className="w-6 h-6 text-blue-600" />
              </div>
              <CardTitle className="text-xl">{title}</CardTitle>
              <span className="text-gray-500 text-sm">
                ({suggestions.length} suggestions)
              </span>
            </div>
            {isExpanded ? (
              <ChevronUp className="w-5 h-5 text-gray-500" />
            ) : (
              <ChevronDown className="w-5 h-5 text-gray-500" />
            )}
          </div>
        </CardHeader>
        
        {isExpanded && (
          <CardContent>
            <div className="space-y-4">
              {suggestions.map((suggestion, index) => (
                <div 
                  key={index} 
                  className="p-4 border rounded-lg hover:shadow-md transition-shadow"
                >
                  <div className="flex justify-between items-start mb-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <AlertCircle className="w-4 h-4 text-blue-500" />
                        <h4 className="font-semibold text-gray-900">
                          {suggestion.suggestion}
                        </h4>
                      </div>
                      <div className="flex flex-wrap gap-2 items-center text-sm text-gray-500">
                        <span className="flex items-center gap-1">
                          <Clock className="w-4 h-4" />
                          {suggestion.timeframe}
                        </span>
                        <span className="px-2">•</span>
                        <span>
                          Difficulty: {suggestion.implementation_difficulty}
                        </span>
                      </div>
                    </div>
                    <div className="flex flex-col items-end gap-2">
                      <PriorityBadge priority={suggestion.priority} />
                      <ImpactBadge value={suggestion.impact_estimate} />
                    </div>
                  </div>
                  {suggestion.metrics && (
                    <div className="bg-gray-50 p-3 rounded-md">
                      <h5 className="text-sm font-medium text-gray-700 mb-2">
                        Key Metrics
                      </h5>
                      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                        {Object.entries(suggestion.metrics).map(([key, value], i) => (
                          <div key={i} className="text-sm">
                            <div className="text-gray-500 capitalize">
                              {key.replace(/_/g, ' ')}
                            </div>
                            <div className="font-medium">
                              {Array.isArray(value) 
                                ? value.join(', ')
                                : typeof value === 'number'
                                  ? value.toLocaleString('en-IN')
                                  : value}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        )}
      </Card>
    );
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">
          Store Optimization Suggestions
        </h1>
        <p className="mt-2 text-gray-600">
          Select a store to view optimization suggestions
        </p>
      </div>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      <div className="bg-white rounded-lg shadow overflow-hidden mb-8">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Store Name
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Category
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Location
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Rating
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {stores.map((store) => (
                <tr key={store.store_id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <Store className="h-5 w-5 text-gray-400 mr-2" />
                      <div className="text-sm font-medium text-gray-900">
                        {store.name}
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {store.category}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {store.location.city}, {store.location.state}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                      {store.ratings.overall}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => fetchSuggestions(store.store_id)}
                      disabled={loading && selectedStore?.store_id === store.store_id}
                    >
                      {loading && selectedStore?.store_id === store.store_id ? (
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      ) : (
                        'View Suggestions'
                      )}
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <Dialog open={!!suggestions} onOpenChange={() => setSuggestions(null)}>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
          {selectedStore && (
            <>
              <DialogHeader>
                <DialogTitle>Suggestions for {selectedStore.name}</DialogTitle>
                <DialogDescription>
                  Optimization suggestions for the next 90 days
                </DialogDescription>
              </DialogHeader>

              <div className="space-y-6 mt-4">
                <SuggestionCard
                  title="Customer Insights"
                  icon={Users}
                  suggestions={suggestions?.customer_suggestions}
                  type="customer"
                />
                <SuggestionCard
                  title="Product Optimization"
                  icon={Package}
                  suggestions={suggestions?.product_suggestions}
                  type="product"
                />
                <SuggestionCard
                  title="Pricing Strategy"
                  icon={DollarSign}
                  suggestions={suggestions?.pricing_suggestions}
                  type="pricing"
                />
                <SuggestionCard
                  title="Timing Optimization"
                  icon={Clock}
                  suggestions={suggestions?.timing_suggestions}
                  type="timing"
                />
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default StoreSuggestionsDashboard;