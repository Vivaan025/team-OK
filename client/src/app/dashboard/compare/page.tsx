'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import StoreComparisonDetailPage from '@/components/ComparisonPage';

export default function ComparisonPage() {
  const [stores, setStores] = useState([]);
  const [selectedStore1, setSelectedStore1] = useState('');
  const [selectedStore2, setSelectedStore2] = useState('');
  const [comparisonData, setComparisonData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch stores list
  useEffect(() => {
    const fetchStores = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/stores');
        if (!response.ok) {
          throw new Error('Failed to fetch stores');
        }
        const data = await response.json();
        setStores(data);
      } catch (err) {
        console.error('Error fetching stores:', err);
        setError('Failed to fetch stores');
      }
    };

    fetchStores();
  }, []);

  // Fetch comparison data when stores are selected
  const fetchComparison = async () => {
    if (!selectedStore1 || !selectedStore2) return;

    setLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `http://localhost:8000/api/compare/${selectedStore1}/${selectedStore2}?days=400`
      );

      if (!response.ok) {
        throw new Error('Failed to fetch store comparison');
      }

      const data = await response.json();
      setComparisonData(data);
    } catch (err) {
      console.error('Error fetching comparison:', err);
      setError('Failed to fetch store comparison');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-6">Store Performance Comparison</h1>
      
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Select Stores to Compare</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="block mb-2 font-medium">First Store</label>
              <select 
                value={selectedStore1}
                onChange={(e) => setSelectedStore1(e.target.value)}
                className="w-full p-2 border rounded"
              >
                <option value="">Select First Store</option>
                {stores.map((store: any) => (
                  <option key={store.store_id} value={store.store_id}>
                    {store.name}
                  </option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block mb-2 font-medium">Second Store</label>
              <select 
                value={selectedStore2}
                onChange={(e) => setSelectedStore2(e.target.value)}
                className="w-full p-2 border rounded"
              >
                <option value="">Select Second Store</option>
                {stores.map((store: any) => (
                  <option key={store.store_id} value={store.store_id}>
                    {store.name}
                  </option>
                ))}
              </select>
            </div>
          </div>
          
          <button 
            onClick={fetchComparison}
            disabled={!selectedStore1 || !selectedStore2}
            className="mt-4 w-full bg-blue-500 text-white p-2 rounded 
              disabled:bg-gray-300 disabled:cursor-not-allowed"
          >
            Compare Stores
          </button>
        </CardContent>
      </Card>

      {loading && (
        <div className="text-center py-6">
          <p>Loading comparison data...</p>
        </div>
      )}

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
          <span className="block sm:inline">{error}</span>
        </div>
      )}

      {comparisonData && (
        <StoreComparisonDetailPage comparisonData={comparisonData} />
      )}
    </div>
  );
}