"use client"
import React, { useState, useEffect, useMemo } from 'react';
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table";
import { 
  Card, 
  CardContent, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { 
  ShoppingCart, 
  CreditCard, 
  Calendar,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle, 
  DialogDescription 
} from "@/components/ui/dialog";

interface TransactionItem {
  product_id: string;
  quantity: number;
  unit_price: number;
  discount: number;
  final_price: number;
  total: number;
}

interface TransactionPayment {
  method: string;
  total_amount: number;
  tax_amount: number;
  final_amount: number;
}

interface Transaction {
  transaction_id: string;
  customer_id: string;
  store_id: string;
  date: string;
  items: TransactionItem[];
  payment: TransactionPayment;
  festivals: string | null;
  loyalty_points_earned: number;
  rating: number | null;
}

const TransactionsListPage: React.FC = () => {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  // Modal state
  const [selectedTransaction, setSelectedTransaction] = useState<Transaction | null>(null);

  useEffect(() => {
    const fetchTransactions = async () => {
      try {
        setIsLoading(true);
        const response = await fetch('http://localhost:8000/api/transactions');
        
        if (!response.ok) {
          throw new Error('Failed to fetch transactions');
        }
        
        const data = await response.json();
        setTransactions(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An unknown error occurred');
      } finally {
        setIsLoading(false);
      }
    };

    fetchTransactions();
  }, []);

  // Memoized pagination logic
  const paginatedTransactions = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage;
    return transactions.slice(startIndex, startIndex + itemsPerPage);
  }, [transactions, currentPage, itemsPerPage]);

  // Calculate total pages
  const totalPages = Math.ceil(transactions.length / itemsPerPage);

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

  const openTransactionDetails = (transaction: Transaction) => {
    setSelectedTransaction(transaction);
  };

  const closeTransactionDetails = () => {
    setSelectedTransaction(null);
  };

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <p>Loading transactions...</p>
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
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center">
              <ShoppingCart className="mr-2" />
              Transactions List
            </div>
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
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Transaction ID</TableHead>
                <TableHead>Date</TableHead>
                <TableHead>Store ID</TableHead>
                <TableHead>Customer ID</TableHead>
                <TableHead>Items Count</TableHead>
                <TableHead>Payment Method</TableHead>
                <TableHead>Total Amount</TableHead>
                <TableHead>Loyalty Points</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {paginatedTransactions.map((transaction) => (
                <TableRow key={transaction.transaction_id}>
                  <TableCell>{transaction.transaction_id}</TableCell>
                  <TableCell>
                    <div className="flex items-center">
                      <Calendar className="mr-2 text-muted-foreground" size={16} />
                      {new Date(transaction.date).toLocaleDateString()}
                    </div>
                  </TableCell>
                  <TableCell>{transaction.store_id}</TableCell>
                  <TableCell>{transaction.customer_id}</TableCell>
                  <TableCell>
                    <Badge variant="secondary">{transaction.items.length}</Badge>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center">
                      <CreditCard className="mr-2 text-muted-foreground" size={16} />
                      {transaction.payment.method}
                    </div>
                  </TableCell>
                  <TableCell>
                    ₹{transaction.payment.final_amount.toFixed(2)}
                  </TableCell>
                  <TableCell>
                    <Badge variant={transaction.loyalty_points_earned > 0 ? "default" : "outline"}>
                      {transaction.loyalty_points_earned}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={() => openTransactionDetails(transaction)}
                    >
                      View Details
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Transaction Details Modal */}
      <Dialog open={!!selectedTransaction} onOpenChange={closeTransactionDetails}>
        <DialogContent className="sm:max-w-[600px]">
          {selectedTransaction && (
            <>
              <DialogHeader>
                <DialogTitle>Transaction Details</DialogTitle>
                <DialogDescription>
                  Detailed information for Transaction ID: {selectedTransaction.transaction_id}
                </DialogDescription>
              </DialogHeader>
              
              <div className="grid gap-4 py-4">
                <div className="grid grid-cols-4 items-center gap-4">
                  <span className="font-medium">Date:</span>
                  <span className="col-span-3">
                    {new Date(selectedTransaction.date).toLocaleString()}
                  </span>
                </div>
                
                <div className="grid grid-cols-4 items-center gap-4">
                  <span className="font-medium">Store ID:</span>
                  <span className="col-span-3">{selectedTransaction.store_id}</span>
                </div>
                
                <div className="grid grid-cols-4 items-center gap-4">
                  <span className="font-medium">Customer ID:</span>
                  <span className="col-span-3">{selectedTransaction.customer_id}</span>
                </div>
                
                <div className="grid grid-cols-4 items-center gap-4">
                  <span className="font-medium">Payment Method:</span>
                  <span className="col-span-3">{selectedTransaction.payment.method}</span>
                </div>
                
                <div className="grid grid-cols-4 items-center gap-4">
                  <span className="font-medium">Loyalty Points:</span>
                  <span className="col-span-3">{selectedTransaction.loyalty_points_earned}</span>
                </div>
                
                <div className="mt-4">
                  <h3 className="text-lg font-semibold mb-2">Items</h3>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Product ID</TableHead>
                        <TableHead>Quantity</TableHead>
                        <TableHead>Unit Price</TableHead>
                        <TableHead>Total</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {selectedTransaction.items.map((item, index) => (
                        <TableRow key={index}>
                          <TableCell>{item.product_id}</TableCell>
                          <TableCell>{item.quantity}</TableCell>
                          <TableCell>₹{item.unit_price.toFixed(2)}</TableCell>
                          <TableCell>₹{item.total.toFixed(2)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
                
                <div className="grid grid-cols-4 items-center gap-4 mt-4">
                  <span className="font-medium">Total Amount:</span>
                  <span className="col-span-3 text-lg font-bold">
                    ₹{selectedTransaction.payment.final_amount.toFixed(2)}
                  </span>
                </div>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default TransactionsListPage;