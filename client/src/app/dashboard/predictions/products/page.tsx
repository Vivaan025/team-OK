"use client";
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { TrendingUp, TrendingDown, Store as StoreIcon, MapPin, Phone, Search } from 'lucide-react';

interface Location {
  state: string;
  city: string;
  address: string;
  pincode: string;
  coordinates: {
    latitude: number;
    longitude: number;
  };
}

interface Contact {
  phone: string;
  email: string;
  manager_name: string;
}

interface Ratings {
  overall: number;
  service: number;
}

interface OpeningHours {
  weekday: string;
  weekend: string;
}

interface Store {
  store_id: string;
  name: string;
  category: string;
  location: Location;
  contact: Contact;
  amenities: string[];
  ratings: Ratings;
  opening_hours: OpeningHours;
  established_date: string;
}

interface Product {
  product_id: string;
  product_name: string;
  current_revenue: number;
  projected_revenue: number;
  growth_potential: number;
  confidence_score: number;
  contributing_factors: string[];
  seasonal_index: number;
  optimal_price: number;
  optimal_stock: number;
}

interface RevenuePrediction {
  store_id: string;
  prediction_period: string;
  predictions: Product[];
  total_projected_increase: number;
  top_categories: string[];
  seasonal_opportunities: any[];
  festival_opportunities: any[];
  confidence_level: number;
}

export default function StoreRevenueDashboard() {
  const [stores, setStores] = useState<Store[]>([]);
  const [selectedStore, setSelectedStore] = useState<Store | null>(null);
  const [revenueData, setRevenueData] = useState<RevenuePrediction | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [months, setMonths] = useState<string>("1");
  const [searchTerm, setSearchTerm] = useState<string>("");

  useEffect(() => {
    fetchStores();
  }, []);

  useEffect(() => {
    if (selectedStore) {
      fetchStoreRevenue(selectedStore.store_id);
    }
  }, [months]);

  const fetchStores = async (): Promise<void> => {
    try {
      const response = await fetch('http://localhost:8000/api/stores');
      const data: Store[] = await response.json();
      setStores(data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching stores:', error);
      setLoading(false);
    }
  };

  const fetchStoreRevenue = async (storeId: string): Promise<void> => {
    try {
      setLoading(true);
      const response = await fetch(`http://localhost:8000/api/revenue-predictions/${storeId}?months=${months}`);
      const data: RevenuePrediction = await response.json();
      setRevenueData(data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching revenue data:', error);
      setLoading(false);
    }
  };

  const formatCurrency = (amount: number): string => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0
    }).format(amount);
  };

  const calculateGrowthPercentage = (current: number, projected: number): string => {
    return ((projected - current) / current * 100).toFixed(1);
  };

  const filteredStores = stores.filter(store => 
    store.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    store.location.city.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleStoreClick = async (store: Store): Promise<void> => {
    setSelectedStore(store);
    await fetchStoreRevenue(store.store_id);
  };

  const renderGrowthTrend = (product: Product) => {
    const growthPercentage = calculateGrowthPercentage(product.current_revenue, product.projected_revenue);
    const isPositive = product.projected_revenue > product.current_revenue;

    return (
      <div className="flex items-center gap-2">
        {isPositive ? (
          <TrendingUp className="w-4 h-4 text-green-500" />
        ) : (
          <TrendingDown className="w-4 h-4 text-red-500" />
        )}
        <span className={`${isPositive ? 'text-green-600' : 'text-red-600'}`}>
          {growthPercentage}%
        </span>
      </div>
    );
  };

  if (loading && !stores.length) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <p>Loading stores...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header with Search */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <h1 className="text-3xl font-bold">Store-Product Analysis</h1>
        <div className="relative w-full sm:w-64">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search stores..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-8"
          />
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold">{stores.length}</div>
            <p className="text-sm text-muted-foreground">Total Stores</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold">
              {(stores.reduce((acc, store) => acc + store.ratings.overall, 0) / stores.length).toFixed(2)}
            </div>
            <p className="text-sm text-muted-foreground">Average Rating</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold">
              {Object.keys(
                stores.reduce((acc, store) => ({ ...acc, [store.category]: true }), {})
              ).length}
            </div>
            <p className="text-sm text-muted-foreground">Store Categories</p>
          </CardContent>
        </Card>
      </div>

      {/* Stores Table */}
      <Card>
        <CardHeader>
          <CardTitle>Stores</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[200px]">Store Name</TableHead>
                  <TableHead>Location</TableHead>
                  <TableHead>Category</TableHead>
                  <TableHead className="text-center">Rating</TableHead>
                  <TableHead className="hidden md:table-cell">Operating Hours</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredStores.map((store) => (
                  <TableRow 
                    key={store.store_id}
                    className="cursor-pointer hover:bg-muted"
                    onClick={() => handleStoreClick(store)}
                  >
                    <TableCell className="font-medium">
                      <div className="flex items-center gap-2">
                        <StoreIcon className="h-4 w-4" />
                        {store.name}
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <MapPin className="h-4 w-4" />
                        {store.location.city}, {store.location.state}
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="secondary">
                        {store.category}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-center">
                      <Badge variant="outline" className="bg-green-50">
                        {store.ratings.overall.toFixed(1)} ⭐
                      </Badge>
                    </TableCell>
                    <TableCell className="hidden md:table-cell">
                      <div className="text-sm text-muted-foreground">
                        {store.opening_hours.weekday}
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      {/* Revenue Details Modal */}
      <Dialog 
        open={Boolean(selectedStore)} 
        onOpenChange={(open) => {
          if (!open) {
            setSelectedStore(null);
            setRevenueData(null);
          }
        }}
      >
        <DialogContent className="max-w-4xl h-[80vh]">
          <DialogHeader>
            <DialogTitle>{selectedStore?.name} - Product Analysis</DialogTitle>
          </DialogHeader>
          
          <div className="flex items-center gap-4 py-4">
            <Select value={months} onValueChange={(value) => {
              setMonths(value);
              if (selectedStore) {
                fetchStoreRevenue(selectedStore.store_id);
              }
            }}>
              <SelectTrigger className="w-40">
                <SelectValue placeholder="Select months" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1">1 Month</SelectItem>
                <SelectItem value="3">3 Months</SelectItem>
                <SelectItem value="6">6 Months</SelectItem>
              </SelectContent>
            </Select>
            
            <Badge variant="outline" className="text-sm">
              {selectedStore?.category}
            </Badge>
            
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Phone className="h-4 w-4" />
              {selectedStore?.contact?.phone}
            </div>
          </div>

          {loading ? (
            <div className="flex items-center justify-center h-40">
              <p>Loading predictions...</p>
            </div>
          ) : revenueData ? (
            <ScrollArea className="h-[calc(80vh-200px)]">
              <div className="space-y-6">
                {/* Summary Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <Card>
                    <CardContent className="pt-6">
                      <div className="text-2xl font-bold">
                        {revenueData.predictions.length}
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Products Analyzed
                      </p>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardContent className="pt-6">
                      <div className="text-2xl font-bold">
                        {formatCurrency(revenueData.total_projected_increase)}
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Projected Increase
                      </p>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardContent className="pt-6">
                      <div className="text-2xl font-bold">
                        {(revenueData.confidence_level * 100).toFixed(1)}%
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Confidence Level
                      </p>
                    </CardContent>
                  </Card>
                </div>

                {/* Products Table */}
                <Card>
                  <CardHeader>
                    <CardTitle>Top Products by Revenue Potential</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="rounded-md border">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Product</TableHead>
                            <TableHead>Current</TableHead>
                            <TableHead>Projected</TableHead>
                            <TableHead>Growth</TableHead>
                            <TableHead>Confidence</TableHead>
                            <TableHead className="hidden md:table-cell">Factors</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {revenueData.predictions.slice(0, 10).map((product) => (
                            <TableRow key={product.product_id}>
                              <TableCell className="font-medium">
                                {product.product_name}
                              </TableCell>
                              <TableCell>
                                {formatCurrency(product.current_revenue)}
                              </TableCell>
                              <TableCell>
                                {formatCurrency(product.projected_revenue)}
                              </TableCell>
                              <TableCell>
                                {renderGrowthTrend(product)}
                              </TableCell>
                              <TableCell>
                                <Progress 
                                  value={product.confidence_score * 100} 
                                  className="w-20"
                                />
                              </TableCell>
                              <TableCell className="hidden md:table-cell">
                                <div className="flex flex-wrap gap-1">
                                  {product.contributing_factors.map((factor, index) => (
                                    <Badge key={index} variant="secondary" className="text-xs">
                                      {factor}
                                    </Badge>
                                  ))}
                                </div>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </ScrollArea>
          ) : null}
        </DialogContent>
      </Dialog>
    </div>
  );
}