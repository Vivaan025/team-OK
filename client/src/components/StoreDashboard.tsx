// Types
interface Coordinates {
    latitude: number;
    longitude: number;
  }
  
  interface Location {
    state: string;
    city: string;
    address: string;
    pincode: string;
    coordinates: Coordinates;
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
  
  interface StoreInfo {
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
  
  interface Forecast {
    date: string;
    sales: number;
    lower_ci: number;
    upper_ci: number;
    is_forecast: boolean;
  }
  
  interface ForecastData {
    forecasts: Forecast[];
  }
  
  // Component
  import React from 'react';
  import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
  import { MapPin, Phone, Mail, Clock, Star } from 'lucide-react';
  
  interface StoreDashboardProps {
    storeInfo: StoreInfo;
    forecastData: ForecastData;
  }
  
  const StoreDashboard: React.FC<StoreDashboardProps> = ({ storeInfo, forecastData }) => {
    return (
      <div className="p-6 space-y-6 bg-white rounded-lg shadow-lg">
        {/* Store Header */}
        <div className="border-b pb-4">
          <h1 className="text-2xl font-bold text-gray-900">{storeInfo.name}</h1>
          <p className="text-gray-600">{storeInfo.category}</p>
        </div>
  
        {/* Store Details Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Location Section */}
          <div className="space-y-3">
            <div className="flex items-start space-x-2">
              <MapPin className="w-5 h-5 text-gray-500 mt-1" />
              <div>
                <p className="text-gray-700">{storeInfo.location.address}</p>
                <p className="text-gray-600">
                  {storeInfo.location.city}, {storeInfo.location.state} - {storeInfo.location.pincode}
                </p>
              </div>
            </div>
  
            {/* Contact Info */}
            <div className="flex items-center space-x-2">
              <Phone className="w-5 h-5 text-gray-500" />
              <p className="text-gray-700">{storeInfo.contact.phone}</p>
            </div>
            <div className="flex items-center space-x-2">
              <Mail className="w-5 h-5 text-gray-500" />
              <p className="text-gray-700">{storeInfo.contact.email}</p>
            </div>
          </div>
  
          {/* Hours and Ratings */}
          <div className="space-y-3">
            <div className="flex items-start space-x-2">
              <Clock className="w-5 h-5 text-gray-500 mt-1" />
              <div>
                <p className="text-gray-700">Weekdays: {storeInfo.opening_hours.weekday}</p>
                <p className="text-gray-700">Weekends: {storeInfo.opening_hours.weekend}</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Star className="w-5 h-5 text-yellow-500" />
              <p className="text-gray-700">Overall: {storeInfo.ratings.overall}/5</p>
            </div>
          </div>
        </div>
  
        {/* Amenities */}
        <div className="border-t pt-4">
          <h2 className="text-lg font-semibold mb-2">Amenities</h2>
          <div className="flex flex-wrap gap-2">
            {storeInfo.amenities.map((amenity) => (
              <span
                key={amenity}
                className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm"
              >
                {amenity}
              </span>
            ))}
          </div>
        </div>
  
        {/* Sales Forecast Chart */}
        <div className="border-t pt-4">
          <h2 className="text-lg font-semibold mb-4">Sales Forecast</h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={forecastData.forecasts}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="date"
                  tickFormatter={(date) => new Date(date).toLocaleDateString()}
                />
                <YAxis />
                <Tooltip
                  labelFormatter={(date) => new Date(date).toLocaleDateString()}
                  formatter={(value) => [`$${value.toLocaleString()}`, 'Sales']}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="sales"
                  stroke="#2563eb"
                  name="Sales"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="upper_ci"
                  stroke="#93c5fd"
                  name="Upper CI"
                  dot={false}
                  strokeDasharray="3 3"
                />
                <Line
                  type="monotone"
                  dataKey="lower_ci"
                  stroke="#93c5fd"
                  name="Lower CI"
                  dot={false}
                  strokeDasharray="3 3"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    );
  };
  
  export default StoreDashboard;