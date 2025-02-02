"use client";
import React, { useState, useEffect, useMemo } from "react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardFooter,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  MapPin,
  Store,
  Phone,
  Mail,
  Star,
  Clock,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface StoreLocation {
  state: string;
  city: string;
  address: string;
  pincode: string;
  coordinates: {
    latitude: number;
    longitude: number;
  };
}

interface StoreContact {
  phone: string;
  email: string;
  manager_name: string;
}

interface StoreRatings {
  overall: number;
  service: number;
}

interface Store {
  store_id: string;
  name: string;
  category: string;
  location: StoreLocation;
  contact: StoreContact;
  amenities: string[];
  ratings: StoreRatings;
  opening_hours: {
    weekday: string;
    weekend: string;
  };
  established_date: string;
}

interface StoreMetrics {
  basic_metrics: {
    total_sales: number;
    total_transactions: number;
    avg_transaction_value: number;
    current_month_sales: number;
    last_month_sales: number;
    mom_growth: number;
  };
  sales_prediction: {
    dates: string[];
    predictions: number[];
    confidence_score: number;
    feature_importance: Record<string, number>;
  };
  customer_insights: {
    total_customers: number;
    avg_customer_spend: number;
    loyal_customers: number;
  };
  product_insights: {
    total_items_sold: number;
    avg_item_price: number;
    top_products: Record<string, number>;
  };
}

interface Product {
  product_id: string;
  name: string;
  category: string;
  subcategory: string;
  brand: string;
  pricing: {
    base_price: number;
    current_price: number;
    discount_percentage: number;
    tax_rate: number;
  };
  inventory: {
    total_stock: number;
    min_stock_threshold: number;
    reorder_quantity: number;
  };
  ratings: {
    average: number;
    count: number;
  };
  specifications: {
    weight: string;
    dimensions: string;
    manufacturer: string;
    country_of_origin: string;
  };
  features: string[];
  launch_date: string;
}

const StoresListPage: React.FC = () => {
  const [stores, setStores] = useState<Store[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 6;

  // Selected store state
  const [selectedStore, setSelectedStore] = useState<Store | null>(null);
  const [storeMetrics, setStoreMetrics] = useState<StoreMetrics | null>(null);
  const [topProducts, setTopProducts] = useState<Product[]>([]);

  useEffect(() => {
    const fetchStores = async () => {
      try {
        setIsLoading(true);
        const response = await fetch("http://localhost:8000/api/stores");

        if (!response.ok) {
          throw new Error("Failed to fetch stores");
        }

        const data = await response.json();
        setStores(data);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "An unknown error occurred"
        );
      } finally {
        setIsLoading(false);
      }
    };

    fetchStores();
  }, []);

  // Fetch store metrics when a store is selected
  const fetchStoreMetrics = async (storeId: string) => {
    try {
      const metricsResponse = await fetch(
        `http://localhost:8000/api/store/${storeId}`
      );

      if (!metricsResponse.ok) {
        throw new Error("Failed to fetch store metrics");
      }

      const metricsData = await metricsResponse.json();

      // Add null checks
      if (
        !metricsData ||
        !metricsData.product_insights ||
        !metricsData.product_insights.top_products
      ) {
        console.error("Invalid metrics data structure", metricsData);
        setStoreMetrics(null);
        setTopProducts([]);
        return;
      }

      setStoreMetrics(metricsData);

      // Fetch top products details
      const topProductPromises = Object.keys(
        metricsData.product_insights.top_products
      ).map(async (productId) => {
        const productResponse = await fetch(
          `http://localhost:8000/api/product/${productId}`
        );

        if (!productResponse.ok) {
          console.error(`Failed to fetch product ${productId}`);
          return null;
        }

        return productResponse.json();
      });

      const topProductsData = await Promise.all(topProductPromises);

      // Filter out any null products
      setTopProducts(topProductsData.filter((product) => product !== null));
    } catch (err) {
      console.error("Failed to fetch store metrics", err);
      setStoreMetrics(null);
      setTopProducts([]);
    }
  };

  // Open store details modal
  const openStoreDetails = (store: Store) => {
    setSelectedStore(store);
    fetchStoreMetrics(store.store_id);
  };

  // Close store details modal
  const closeStoreDetails = () => {
    setSelectedStore(null);
    setStoreMetrics(null);
    setTopProducts([]);
  };

  // Pagination logic
  const paginatedStores = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage;
    return stores.slice(startIndex, startIndex + itemsPerPage);
  }, [stores, currentPage]);

  const totalPages = Math.ceil(stores.length / itemsPerPage);

  const handlePreviousPage = () => {
    if (currentPage > 1) {
      setCurrentPage(currentPage - 1);
    }
  };

  const handleNextPage = () => {
    if (currentPage < totalPages) {
      setCurrentPage(currentPage + 1);
    }
  };

  // Prepare sales prediction chart data
  const salesPredictionChartData = useMemo(() => {
    if (!storeMetrics) return [];

    return storeMetrics.sales_prediction.dates.map((date, index) => ({
      date: new Date(date).toLocaleDateString(),
      prediction: storeMetrics.sales_prediction.predictions[index],
    }));
  }, [storeMetrics]);

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <p>Loading stores...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex justify-center items-center h-screen text-red-500">
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Our Stores</h1>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handlePreviousPage}
            disabled={currentPage === 1}
          >
            <ChevronLeft className="mr-1" size={16} /> Previous
          </Button>
          <span className="text-sm">
            Page {currentPage} of {totalPages}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={handleNextPage}
            disabled={currentPage === totalPages}
          >
            Next <ChevronRight className="ml-1" size={16} />
          </Button>
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {paginatedStores.map((store) => (
          <Card
            key={store.store_id}
            className="hover:shadow-lg transition-shadow"
          >
            <CardHeader>
              <CardTitle className="flex justify-between items-center">
                <span>{store.name}</span>
                <Badge variant="secondary">{store.category}</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center">
                  <MapPin className="mr-2 text-muted-foreground" size={20} />
                  <span>{`${store.location.address}, ${store.location.city}, ${store.location.state}`}</span>
                </div>
                <div className="flex items-center">
                  <Phone className="mr-2 text-muted-foreground" size={20} />
                  <span>{store.contact.phone}</span>
                </div>
                <div className="flex items-center">
                  <Mail className="mr-2 text-muted-foreground" size={20} />
                  <span>{store.contact.email}</span>
                </div>
                <div className="flex items-center">
                  <Star className="mr-2 text-muted-foreground" size={20} />
                  <span>Overall Rating: {store.ratings.overall}/5</span>
                </div>
                <div className="flex items-center">
                  <Clock className="mr-2 text-muted-foreground" size={20} />
                  <span>Weekdays: {store.opening_hours.weekday}</span>
                </div>
                <div className="flex flex-wrap gap-2 mt-2">
                  {store.amenities.map((amenity) => (
                    <Badge key={amenity} variant="outline">
                      {amenity}
                    </Badge>
                  ))}
                </div>
              </div>
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button variant="outline" onClick={() => openStoreDetails(store)}>
                View Details
              </Button>
              <Button>Contact Manager</Button>
            </CardFooter>
          </Card>
        ))}
      </div>

      {/* Store Details Modal */}
      <Dialog open={!!selectedStore} onOpenChange={closeStoreDetails}>
        <DialogContent className="sm:max-w-[800px] max-h-[90vh] overflow-y-auto">
          <DialogTitle>Store Details</DialogTitle>

          {selectedStore && storeMetrics && (
            <>
              <DialogHeader>
                <DialogTitle>{selectedStore.name} - Store Details</DialogTitle>
                <DialogDescription>
                  Comprehensive store performance insights
                </DialogDescription>
              </DialogHeader>

              <div className="grid gap-6">
                {/* Basic Metrics Section */}
                <Card>
                  <CardHeader>
                    <CardTitle>Basic Metrics</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      <div>
                        <p className="text-muted-foreground">Total Sales</p>
                        <p className="font-bold">
                          ₹{storeMetrics.basic_metrics.total_sales.toFixed(2)}
                        </p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">
                          Total Transactions
                        </p>
                        <p className="font-bold">
                          {storeMetrics.basic_metrics.total_transactions}
                        </p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">
                          Avg Transaction Value
                        </p>
                        <p className="font-bold">
                          ₹
                          {storeMetrics.basic_metrics.avg_transaction_value.toFixed(
                            2
                          )}
                        </p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">
                          Current Month Sales
                        </p>
                        <p className="font-bold">
                          ₹
                          {storeMetrics.basic_metrics.current_month_sales.toFixed(
                            2
                          )}
                        </p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">
                          Last Month Sales
                        </p>
                        <p className="font-bold">
                          ₹
                          {storeMetrics.basic_metrics.last_month_sales.toFixed(
                            2
                          )}
                        </p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">MoM Growth</p>
                        <p
                          className={`font-bold ${
                            storeMetrics.basic_metrics.mom_growth >= 0
                              ? "text-green-600"
                              : "text-red-600"
                          }`}
                        >
                          {storeMetrics.basic_metrics.mom_growth.toFixed(2)}%
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Sales Prediction Chart */}
                <Card>
                  <CardHeader>
                    <CardTitle>Sales Prediction</CardTitle>
                    <p className="text-muted-foreground">
                      Confidence Score:{" "}
                      {(
                        storeMetrics.sales_prediction.confidence_score * 100
                      ).toFixed(2)}
                      %
                    </p>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={salesPredictionChartData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="prediction"
                          stroke="#8884d8"
                          activeDot={{ r: 8 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>

                {/* Customer Insights */}
                <Card>
                  <CardHeader>
                    <CardTitle>Customer Insights</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-2 gap-4">
                      <div>
                        <p className="text-muted-foreground">Total Customers</p>
                        <p className="font-bold">
                          {storeMetrics.customer_insights.total_customers}
                        </p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">
                          Avg Customer Spend
                        </p>
                        <p className="font-bold">
                          ₹
                          {storeMetrics.customer_insights.avg_customer_spend.toFixed(
                            2
                          )}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Top Products */}
                <Card>
                  <CardHeader>
                    <CardTitle>Top Products</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {topProducts.length > 0 ? (
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Product Name</TableHead>
                            <TableHead>Category</TableHead>
                            <TableHead>Quantity Sold</TableHead>
                            <TableHead>Price</TableHead>
                            <TableHead>Rating</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {topProducts.map((product) => (
                            <TableRow key={product.product_id}>
                              <TableCell>{product.name}</TableCell>
                              <TableCell>
                                <Badge variant="secondary">
                                  {product.category}
                                </Badge>
                              </TableCell>
                              <TableCell>
                                {storeMetrics?.product_insights?.top_products?.[
                                  product.product_id
                                ] || "N/A"}
                              </TableCell>
                              <TableCell>
                                ₹{product.pricing.current_price.toFixed(2)}
                              </TableCell>
                              <TableCell>
                                <div className="flex items-center">
                                  <Star
                                    className="mr-1 text-yellow-500"
                                    size={16}
                                  />
                                  {product.ratings.average.toFixed(1)}/5
                                  <span className="text-muted-foreground ml-2">
                                    ({product.ratings.count} reviews)
                                  </span>
                                </div>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    ) : (
                      <p className="text-muted-foreground">
                        No top products found
                      </p>
                    )}
                  </CardContent>
                </Card>

                {/* Product Insights */}
                <Card>
                  <CardHeader>
                    <CardTitle>Product Insights</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      <div>
                        <p className="text-muted-foreground">
                          Total Items Sold
                        </p>
                        <p className="font-bold">
                          {storeMetrics.product_insights.total_items_sold}
                        </p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Avg Item Price</p>
                        <p className="font-bold">
                          ₹
                          {storeMetrics.product_insights.avg_item_price.toFixed(
                            2
                          )}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default StoresListPage;
