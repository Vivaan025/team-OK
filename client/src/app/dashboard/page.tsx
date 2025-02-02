"use client"
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table";
import { 
  DollarSign, 
  ShoppingBag, 
  Users, 
  Store, 
  IndianRupee
} from 'lucide-react';

// Type Definitions
interface Transaction {
  transaction_id: string;
  customer_id: string;
  store_id: string;
  date: string;
  items: {
    product_id: string;
    quantity: number;
    unit_price: number;
    discount: number;
    final_price: number;
    total: number;
  }[];
  payment: {
    method: string;
    total_amount: number;
    tax_amount: number;
    final_amount: number;
  };
  loyalty_points_earned: number;
}

interface Store {
  store_id: string;
  name: string;
  category: string;
  location: {
    state: string;
    city: string;
    address: string;
    pincode: string;
  };
  contact: {
    phone: string;
    email: string;
    manager_name: string;
  };
  ratings: {
    overall: number;
    service: number;
  };
}

interface Customer {
  customer_id: string;
  personal_info: {
    name: string;
    age: number;
    gender: string;
    email: string;
  };
  loyalty_info: {
    membership_tier: string;
    points: number;
  };
}

interface DashboardStats {
  totalRevenue: number;
  totalTransactions: number;
  totalCustomers: number;
  totalStores: number;
}

interface RevenueByStoreData {
  storeId: string;
  storeName: string;
  revenue: number;
}

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<DashboardStats>({
    totalRevenue: 0,
    totalTransactions: 0,
    totalCustomers: 0,
    totalStores: 0
  });
  const [transactionsData, setTransactionsData] = useState<Transaction[]>([]);
  const [storesData, setStoresData] = useState<Store[]>([]);
  const [customersData, setCustomersData] = useState<Customer[]>([]);
  const [revenueByStoreData, setRevenueByStoreData] = useState<RevenueByStoreData[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        // Simulated data fetching - replace with actual API calls
        const [transactionsResponse, storesResponse, customersResponse] = await Promise.all([
          fetch('http://localhost:8000/api/transactions'),
          fetch('http://localhost:8000/api/stores'),
          fetch('http://localhost:8000/api/customers')
        ]);

        const transactions: Transaction[] = await transactionsResponse.json();
        const stores: Store[] = await storesResponse.json();
        const customers: Customer[] = await customersResponse.json();

        // Calculate total revenue
        const totalRevenue = transactions.reduce((sum, transaction) => 
          sum + transaction.payment.final_amount, 0);

        // Prepare data for bar chart
        const revenueByStore = transactions.reduce((acc: Record<string, number>, transaction) => {
          const storeId = transaction.store_id;
          acc[storeId] = (acc[storeId] || 0) + transaction.payment.final_amount;
          return acc;
        }, {});

        const barChartData: RevenueByStoreData[] = Object.entries(revenueByStore).map(([storeId, revenue]) => ({
          storeId,
          storeName: stores.find((s) => s.store_id === storeId)?.name || 'Unknown',
          revenue: Math.round(revenue)
        }));

        setStats({
          totalRevenue: Math.round(totalRevenue),
          totalTransactions: transactions.length,
          totalCustomers: customers.length,
          totalStores: stores.length
        });

        setTransactionsData(transactions);
        setStoresData(stores);
        setCustomersData(customers);
        setRevenueByStoreData(barChartData);
        setIsLoading(false);
      } catch (error) {
        console.error('Error fetching data:', error);
        setError('Failed to fetch dashboard data');
        setIsLoading(false);
      }
    };

    fetchData();
  }, []);

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <p>Loading dashboard...</p>
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
    <div className="p-4 space-y-4">
      <h1 className="text-2xl font-bold mb-4">Dashboard</h1>
      
      {/* Key Performance Indicators */}
      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Revenue</CardTitle>
            <IndianRupee className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">₹{stats.totalRevenue.toLocaleString()}</div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Transactions</CardTitle>
            <ShoppingBag className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.totalTransactions}</div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Customers</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.totalCustomers}</div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Stores</CardTitle>
            <Store className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.totalStores}</div>
          </CardContent>
        </Card>
      </div>

      {/* Revenue by Store Chart */}
      <Card className="mt-4">
        <CardHeader>
          <CardTitle>Revenue by Store</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={revenueByStoreData}>
              <XAxis dataKey="storeName" />
              <YAxis />
              <Tooltip 
                formatter={(value: number) => [`₹${value.toLocaleString()}`, 'Revenue']}
              />
              <Legend />
              <Bar dataKey="revenue" fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Recent Transactions */}
      <Card className="mt-4">
        <CardHeader>
          <CardTitle>Recent Transactions</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Transaction ID</TableHead>
                <TableHead>Store</TableHead>
                <TableHead>Customer</TableHead>
                <TableHead>Total Amount</TableHead>
                <TableHead>Payment Method</TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {transactionsData.slice(0, 5).map((transaction) => (
                <TableRow key={transaction.transaction_id}>
                  <TableCell>{transaction.transaction_id.slice(0, 8)}</TableCell>
                  <TableCell>{storesData.find((s) => s.store_id === transaction.store_id)?.name || 'Unknown'}</TableCell>
                  <TableCell>{transaction.customer_id.slice(0, 8)}</TableCell>
                  <TableCell>₹{transaction.payment.final_amount.toLocaleString()}</TableCell>
                  <TableCell>{transaction.payment.method}</TableCell>
                  <TableCell>
                    <Badge variant="outline">Completed</Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
};

export default Dashboard;