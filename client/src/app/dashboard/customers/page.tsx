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
  Users, 
  User,
  Mail, 
  Phone, 
  MapPin,
  ChevronLeft,
  ChevronRight,
  CreditCard,
  Star
} from 'lucide-react';
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle, 
  DialogDescription 
} from "@/components/ui/dialog";
import { StorePreferencePrediction } from '@/components/PreferencePrediction';

interface CustomerAddress {
  street: string;
  city: string;
  state: string;
  pincode: string;
}

interface CustomerPersonalInfo {
  name: string;
  age: number;
  gender: string;
  email: string;
  phone: string;
  address: CustomerAddress;
}

interface ShoppingPreferences {
  preferred_payment_methods: string[];
  preferred_categories: string[];
  preferred_shopping_times: string;
}

interface LoyaltyInfo {
  membership_tier: string;
  points: number;
  member_since: string;
}

interface Customer {
  customer_id: string;
  personal_info: CustomerPersonalInfo;
  shopping_preferences: ShoppingPreferences;
  loyalty_info: LoyaltyInfo;
}

const CustomersListPage: React.FC = () => {
  const [customers, setCustomers] = useState<Customer[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  // Selected customer state
  const [selectedCustomer, setSelectedCustomer] = useState<Customer | null>(null);

  useEffect(() => {
    const fetchCustomers = async () => {
      try {
        setIsLoading(true);
        const response = await fetch('http://localhost:8000/api/customers');
        
        if (!response.ok) {
          throw new Error('Failed to fetch customers');
        }
        
        const data = await response.json();
        setCustomers(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An unknown error occurred');
      } finally {
        setIsLoading(false);
      }
    };

    fetchCustomers();
  }, []);

  // Pagination logic
  const paginatedCustomers = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage;
    return customers.slice(startIndex, startIndex + itemsPerPage);
  }, [customers, currentPage]);

  const totalPages = Math.ceil(customers.length / itemsPerPage);

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

  // Open customer details modal
  const openCustomerDetails = (customer: Customer) => {
    setSelectedCustomer(customer);
  };

  // Close customer details modal
  const closeCustomerDetails = () => {
    setSelectedCustomer(null);
  };

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <p>Loading customers...</p>
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
        <h1 className="text-3xl font-bold flex items-center">
          <Users className="mr-2" /> Customers
        </h1>
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

      <Card>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Email</TableHead>
                <TableHead>City</TableHead>
                <TableHead>Membership Tier</TableHead>
                <TableHead>Loyalty Points</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {paginatedCustomers.map((customer) => (
                <TableRow key={customer.customer_id}>
                  {/* <TableCell>{customer.customer_id}</TableCell> */}
                  <TableCell>
                    <div className="flex items-center">
                      <User className="mr-2 text-muted-foreground" size={16} />
                      {customer.personal_info.name}
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center">
                      <Mail className="mr-2 text-muted-foreground" size={16} />
                      {customer.personal_info.email}
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center">
                      <MapPin className="mr-2 text-muted-foreground" size={16} />
                      {customer.personal_info.address.city}
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge variant={
                      customer.loyalty_info.membership_tier === 'Gold' ? 'default' :
                      customer.loyalty_info.membership_tier === 'Silver' ? 'secondary' : 'outline'
                    }>
                      {customer.loyalty_info.membership_tier}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center">
                      <Star className="mr-1 text-yellow-500" size={16} />
                      {customer.loyalty_info.points}
                    </div>
                  </TableCell>
                  <TableCell>
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => openCustomerDetails(customer)}
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

      {/* Customer Details Modal */}
      <Dialog open={!!selectedCustomer} onOpenChange={closeCustomerDetails}>
        <DialogContent className="sm:max-w-[800px] max-h-[90vh] overflow-y-auto">
          {selectedCustomer && (
            <>
              <DialogHeader>
                <DialogTitle>Customer Details</DialogTitle>
                <DialogDescription>
                  Comprehensive information for {selectedCustomer.personal_info.name}
                </DialogDescription>
              </DialogHeader>
              
              <div className="grid gap-6">
                {/* Personal Information */}
                <Card>
                  <CardHeader>
                    <CardTitle>Personal Information</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-muted-foreground">Name</p>
                        <p className="font-bold">{selectedCustomer.personal_info.name}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Age</p>
                        <p className="font-bold">{selectedCustomer.personal_info.age}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Gender</p>
                        <p className="font-bold">{selectedCustomer.personal_info.gender}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Email</p>
                        <p className="font-bold">{selectedCustomer.personal_info.email}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Phone</p>
                        <p className="font-bold">{selectedCustomer.personal_info.phone}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Address */}
                <Card>
                  <CardHeader>
                    <CardTitle>Address</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                      <p className="text-muted-foreground">Street</p>
                        <p className="font-bold">{selectedCustomer.personal_info.address.street}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">City</p>
                        <p className="font-bold">{selectedCustomer.personal_info.address.city}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">State</p>
                        <p className="font-bold">{selectedCustomer.personal_info.address.state}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Pincode</p>
                        <p className="font-bold">{selectedCustomer.personal_info.address.pincode}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Shopping Preferences */}
                <Card>
                  <CardHeader>
                    <CardTitle>Shopping Preferences</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-muted-foreground">Payment Methods</p>
                        <div className="flex flex-wrap gap-2">
                          {selectedCustomer.shopping_preferences.preferred_payment_methods.map((method) => (
                            <Badge key={method} variant="outline">{method}</Badge>
                          ))}
                        </div>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Preferred Categories</p>
                        <div className="flex flex-wrap gap-2">
                          {selectedCustomer.shopping_preferences.preferred_categories.map((category) => (
                            <Badge key={category} variant="secondary">{category}</Badge>
                          ))}
                        </div>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Preferred Shopping Times</p>
                        <p className="font-bold">{selectedCustomer.shopping_preferences.preferred_shopping_times}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Loyalty Information */}
                <Card>
                  <CardHeader>
                    <CardTitle>Loyalty Information</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-muted-foreground">Membership Tier</p>
                        <Badge variant={
                          selectedCustomer.loyalty_info.membership_tier === 'Gold' ? 'default' :
                          selectedCustomer.loyalty_info.membership_tier === 'Silver' ? 'secondary' : 'outline'
                        }>
                          {selectedCustomer.loyalty_info.membership_tier}
                        </Badge>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Loyalty Points</p>
                        <div className="flex items-center font-bold">
                          <Star className="mr-1 text-yellow-500" size={16} />
                          {selectedCustomer.loyalty_info.points}
                        </div>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Member Since</p>
                        <p className="font-bold">
                          {new Date(selectedCustomer.loyalty_info.member_since).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Store Preference Prediction */}
                <StorePreferencePrediction 
                  customerId={selectedCustomer.customer_id} 
                />
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default CustomersListPage;