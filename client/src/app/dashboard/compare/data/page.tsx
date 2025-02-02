import React from "react";
import { Bar, Line, Pie } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

// Utility function to format currency
const formatCurrency = (value: number) => {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
};

// Revenue Comparison Component
export const RevenueComparison: React.FC<{ comparisonData: any }> = ({
  comparisonData,
}) => {
  const revenueData = [
    {
      name: comparisonData.store1_name,
      revenue: comparisonData.revenue_comparison.store1_value,
    },
    {
      name: comparisonData.store2_name,
      revenue: comparisonData.revenue_comparison.store2_value,
    },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Revenue Comparison</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <h4 className="font-semibold">{comparisonData.store1_name}</h4>
            <p className="text-lg">
              {formatCurrency(comparisonData.revenue_comparison.store1_value)}
            </p>
          </div>
          <div>
            <h4 className="font-semibold">{comparisonData.store2_name}</h4>
            <p className="text-lg">
              {formatCurrency(comparisonData.revenue_comparison.store2_value)}
            </p>
          </div>
        </div>
        <p className="mt-4 text-sm text-gray-600">
          {comparisonData.revenue_comparison.insight}
        </p>
      </CardContent>
    </Card>
  );
};

// Product Category Performance Component
export const CategoryPerformance: React.FC<{ comparisonData: any }> = ({
  comparisonData,
}) => {
  const categoryData = Object.entries(comparisonData.product_comparison).map(
    ([category, comparison]) => ({
      category,
      store1Value: comparisonData.store1_value,
      store2Value: comparisonData.store2_value,
    })
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle>Category Performance</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-gray-100">
                <th className="p-2 text-left">Category</th>
                <th className="p-2 text-right">{comparisonData.store1_name}</th>
                <th className="p-2 text-right">{comparisonData.store2_name}</th>
                <th className="p-2 text-center">Insight</th>
              </tr>
            </thead>
            <tbody>
              {categoryData.map((item) => (
                <tr key={item.category} className="border-b">
                  <td className="p-2">{item.category}</td>
                  <td className="p-2 text-right">
                    {formatCurrency(item.store1Value)}
                  </td>
                  <td className="p-2 text-right">
                    {formatCurrency(item.store2Value)}
                  </td>
                  <td className="p-2 text-center text-sm text-gray-600">
                    {comparisonData.product_comparison[item.category].insight}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
};

// Transaction Metrics Comparison
export const TransactionComparison: React.FC<{ comparisonData: any }> = ({
  comparisonData,
}) => {
  const metrics1 = comparisonData.transaction_metrics_1;
  const metrics2 = comparisonData.transaction_metrics_2;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Transaction Metrics</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <h4 className="font-semibold">{comparisonData.store1_name}</h4>
            <ul className="space-y-2">
              <li>
                <strong>Total Transactions:</strong>{" "}
                {metrics1.total_transactions}
              </li>
              <li>
                <strong>Avg Transaction Value:</strong>{" "}
                {formatCurrency(metrics1.avg_transaction_value)}
              </li>
              <li>
                <strong>Peak Hours:</strong> {metrics1.peak_hours.join(", ")}
              </li>
              <li>
                <strong>Popular Payment Methods:</strong>{" "}
                {metrics1.popular_payment_methods.join(", ")}
              </li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold">{comparisonData.store2_name}</h4>
            <ul className="space-y-2">
              <li>
                <strong>Total Transactions:</strong>{" "}
                {metrics2.total_transactions}
              </li>
              <li>
                <strong>Avg Transaction Value:</strong>{" "}
                {formatCurrency(metrics2.avg_transaction_value)}
              </li>
              <li>
                <strong>Peak Hours:</strong> {metrics2.peak_hours.join(", ")}
              </li>
              <li>
                <strong>Popular Payment Methods:</strong>{" "}
                {metrics2.popular_payment_methods.join(", ")}
              </li>
            </ul>
          </div>
        </div>
        <p className="mt-4 text-sm text-gray-600">
          {comparisonData.transaction_comparison.insight}
        </p>
      </CardContent>
    </Card>
  );
};

// Customer Metrics Comparison
export const CustomerComparison: React.FC<{ comparisonData: any }> = ({
  comparisonData,
}) => {
  const metrics1 = comparisonData.customer_metrics_1;
  const metrics2 = comparisonData.customer_metrics_2;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Customer Metrics</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <h4 className="font-semibold">{comparisonData.store1_name}</h4>
            <ul className="space-y-2">
              <li>
                <strong>Total Customers:</strong> {metrics1.total_customers}
              </li>
              <li>
                <strong>Repeat Customers:</strong> {metrics1.repeat_customers}
              </li>
              <li>
                <strong>Avg Customer Lifetime Value:</strong>{" "}
                {formatCurrency(metrics1.avg_customer_lifetime_value)}
              </li>
              <li>
                <strong>Customer Satisfaction:</strong>{" "}
                {metrics1.customer_satisfaction?.toFixed(2) || "N/A"}
              </li>
              <li>
                <strong>Membership Distribution:</strong>
                {Object.entries(metrics1.membership_distribution)
                  .map(([tier, count]) => `${tier}: ${count}`)
                  .join(", ")}
              </li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold">{comparisonData.store2_name}</h4>
            <ul className="space-y-2">
              <li>
                <strong>Total Customers:</strong> {metrics2.total_customers}
              </li>
              <li>
                <strong>Repeat Customers:</strong> {metrics2.repeat_customers}
              </li>
              <li>
                <strong>Avg Customer Lifetime Value:</strong>{" "}
                {formatCurrency(metrics2.avg_customer_lifetime_value)}
              </li>
              <li>
                <strong>Customer Satisfaction:</strong>{" "}
                {metrics2.customer_satisfaction?.toFixed(2) || "N/A"}
              </li>
              <li>
                <strong>Membership Distribution:</strong>
                {Object.entries(metrics2.membership_distribution)
                  .map(([tier, count]) => `${tier}: ${count}`)
                  .join(", ")}
              </li>
            </ul>
          </div>
        </div>
        <div className="mt-4 space-y-2">
          {
            Object.entries(comparisonData.customer_comparison)
              .filter(([key]) => key !== "membership_distribution")
              .map(([key, comparison]) => {
                // Check if comparison is an object with an 'insight' property
                if (
                  typeof comparison === "object" &&
                  comparison !== null &&
                  "insight" in comparison
                ) {
                  return (
                    <p key={key} className="text-sm text-gray-600">
                      {(comparison as { insight: string }).insight}
                    </p>
                  );
                }
                return null;
              })
              .filter(Boolean) // Remove any null values
          }
        </div>
      </CardContent>
    </Card>
  );
};

// Operational Metrics Comparison
export const OperationalComparison: React.FC<{ comparisonData: any }> = ({
  comparisonData,
}) => {
  const metrics1 = comparisonData.operational_metrics_1;
  const metrics2 = comparisonData.operational_metrics_2;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Operational Metrics</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <h4 className="font-semibold">{comparisonData.store1_name}</h4>
            <ul className="space-y-2">
              <li>
                <strong>Amenities:</strong> {metrics1.amenities.join(", ")}
              </li>
              <li>
                <strong>Operating Hours (Weekday):</strong>{" "}
                {metrics1.operating_hours.weekday}
              </li>
              <li>
                <strong>Operating Hours (Weekend):</strong>{" "}
                {metrics1.operating_hours.weekend}
              </li>
              <li>
                <strong>Parking Available:</strong>{" "}
                {metrics1.parking_available ? "Yes" : "No"}
              </li>
              <li>
                <strong>Unique Advantages:</strong>{" "}
                {comparisonData.unique_advantages_1.join(", ")}
              </li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold">{comparisonData.store2_name}</h4>
            <ul className="space-y-2">
              <li>
                <strong>Amenities:</strong> {metrics2.amenities.join(", ")}
              </li>
              <li>
                <strong>Operating Hours (Weekday):</strong>{" "}
                {metrics2.operating_hours.weekday}
              </li>
              <li>
                <strong>Operating Hours (Weekend):</strong>{" "}
                {metrics2.operating_hours.weekend}
              </li>
              <li>
                <strong>Parking Available:</strong>{" "}
                {metrics2.parking_available ? "Yes" : "No"}
              </li>
              <li>
                <strong>Unique Advantages:</strong>{" "}
                {comparisonData.unique_advantages_2.join(", ")}
              </li>
            </ul>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

// Improvement Suggestions Component
export const ImprovementSuggestions: React.FC<{ comparisonData: any }> = ({
  comparisonData,
}) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Improvement Suggestions</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <h4 className="font-semibold">{comparisonData.store1_name}</h4>
            <ul className="list-disc pl-5 space-y-2">
              {comparisonData.improvement_suggestions[comparisonData.store1_id]
                .length > 0 ? (
                comparisonData.improvement_suggestions[
                  comparisonData.store1_id
                ].map((suggestion: string, index: number) => (
                  <li key={index} className="text-sm">
                    {suggestion}
                  </li>
                ))
              ) : (
                <li className="text-sm text-gray-500">
                  No specific improvement suggestions
                </li>
              )}
            </ul>
          </div>
          <div>
            <h4 className="font-semibold">{comparisonData.store2_name}</h4>
            <ul className="list-disc pl-5 space-y-2">
              {comparisonData.improvement_suggestions[comparisonData.store2_id]
                .length > 0 ? (
                comparisonData.improvement_suggestions[
                  comparisonData.store2_id
                ].map((suggestion: string, index: number) => (
                  <li key={index} className="text-sm">
                    {suggestion}
                  </li>
                ))
              ) : (
                <li className="text-sm text-gray-500">
                  No specific improvement suggestions
                </li>
              )}
            </ul>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

// Main Comparison Page
export const StoreComparisonDetailPage: React.FC<{ comparisonData: any }> = ({
  comparisonData,
}) => {
  return (
    <div className="container mx-auto p-4 space-y-6">
      <h1 className="text-2xl font-bold mb-6">
        Store Comparison: {comparisonData.store1_name} vs{" "}
        {comparisonData.store2_name}
      </h1>

      <div className="grid md:grid-cols-2 gap-6">
        <RevenueComparison comparisonData={comparisonData} />
        <CategoryPerformance comparisonData={comparisonData} />
      </div>

      <TransactionComparison comparisonData={comparisonData} />

      <CustomerComparison comparisonData={comparisonData} />

      <OperationalComparison comparisonData={comparisonData} />

      <ImprovementSuggestions comparisonData={comparisonData} />
    </div>
  );
};

export default StoreComparisonDetailPage;
