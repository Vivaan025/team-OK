'use client'
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ChevronDown, ChevronUp, Store } from 'lucide-react';

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

interface ForecastDataPoint {
  date: string;
  sales: number;
  lower_ci: number;
  upper_ci: number;
}

interface ForecastSummary {
  mean_forecast: number;
  min_forecast: number;
  max_forecast: number;
  total_forecast: number;
}

interface ForecastResponse {
  forecasts: ForecastDataPoint[];
  summary: ForecastSummary;
}

const StoreForecast: React.FC = () => {
  const [stores, setStores] = useState<StoreData[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedStore, setSelectedStore] = useState<string | null>(null);
  const [forecastData, setForecastData] = useState<ForecastResponse | null>(null);
  const [expandedRow, setExpandedRow] = useState<string | null>(null);

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
      setError('Failed to load stores. Please try again later.');
      console.error('Error:', err);
    }
  };

  const fetchForecast = async (storeId: string): Promise<void> => {
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/api/forecast/${storeId}?days=30`);
      if (!response.ok) throw new Error('Failed to fetch forecast');
      const data: ForecastResponse = await response.json();
      setForecastData(data);
      setSelectedStore(storeId);
      setExpandedRow(storeId);
    } catch (err) {
      setError('Failed to load forecast. Please try again later.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (value: number): string => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0
    }).format(value);
  };

  const formatDate = (dateString: string): string => {
    return new Date(dateString).toLocaleDateString('en-IN');
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-2xl font-bold">Store Forecast Dashboard</CardTitle>
        </CardHeader>
      </Card>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Store Name</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Category</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Location</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Rating</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {stores.map((store) => (
                <React.Fragment key={store.store_id}>
                  <tr className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <Store className="h-5 w-5 text-gray-400 mr-2" />
                        <div className="text-sm font-medium text-gray-900">{store.name}</div>
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
                        onClick={() => fetchForecast(store.store_id)}
                        disabled={loading}
                        className="inline-flex items-center"
                      >
                        {expandedRow === store.store_id ? <ChevronUp className="h-4 w-4 mr-1" /> : <ChevronDown className="h-4 w-4 mr-1" />}
                        {loading && selectedStore === store.store_id ? 'Loading...' : 'Forecast'}
                      </Button>
                    </td>
                  </tr>
                  {expandedRow === store.store_id && forecastData && (
                    <tr>
                      <td colSpan={5} className="px-6 py-4">
                        <Card>
                          <CardHeader>
                            <CardTitle>Sales Forecast - {store.name}</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="h-96">
                              <ResponsiveContainer width="100%" height="100%">
                                <LineChart
                                  data={forecastData.forecasts}
                                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                                >
                                  <CartesianGrid strokeDasharray="3 3" />
                                  <XAxis 
                                    dataKey="date" 
                                    tickFormatter={formatDate}
                                    angle={-45}
                                    textAnchor="end"
                                  />
                                  <YAxis tickFormatter={formatCurrency} />
                                  <Tooltip 
                                    formatter={(value: number) => formatCurrency(value)}
                                    labelFormatter={formatDate}
                                  />
                                  <Legend />
                                  <Line
                                    type="monotone"
                                    dataKey="sales"
                                    stroke="#8884d8"
                                    name="Sales"
                                    strokeWidth={2}
                                  />
                                  <Line
                                    type="monotone"
                                    dataKey="lower_ci"
                                    stroke="#82ca9d"
                                    strokeDasharray="3 3"
                                    name="Lower Bound"
                                  />
                                  <Line
                                    type="monotone"
                                    dataKey="upper_ci"
                                    stroke="#ffc658"
                                    strokeDasharray="3 3"
                                    name="Upper Bound"
                                  />
                                </LineChart>
                              </ResponsiveContainer>
                            </div>
                            <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
                              <div className="bg-gray-50 p-4 rounded-lg">
                                <h4 className="text-sm font-medium text-gray-500">Mean Forecast</h4>
                                <p className="mt-1 text-lg font-semibold">{formatCurrency(forecastData.summary.mean_forecast)}</p>
                              </div>
                              <div className="bg-gray-50 p-4 rounded-lg">
                                <h4 className="text-sm font-medium text-gray-500">Min Forecast</h4>
                                <p className="mt-1 text-lg font-semibold">{formatCurrency(forecastData.summary.min_forecast)}</p>
                              </div>
                              <div className="bg-gray-50 p-4 rounded-lg">
                                <h4 className="text-sm font-medium text-gray-500">Max Forecast</h4>
                                <p className="mt-1 text-lg font-semibold">{formatCurrency(forecastData.summary.max_forecast)}</p>
                              </div>
                              <div className="bg-gray-50 p-4 rounded-lg">
                                <h4 className="text-sm font-medium text-gray-500">Total Forecast</h4>
                                <p className="mt-1 text-lg font-semibold">{formatCurrency(forecastData.summary.total_forecast)}</p>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default StoreForecast;