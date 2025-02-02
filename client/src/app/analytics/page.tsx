import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import LineChart from "@/components/charts/linechart";
import BarChart from "@/components/charts/barchart";
import PieChart from "@/components/charts/PieCharts";

const DashboardCharts: React.FC = () => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* Line Chart */}
      <Card className="p-4 shadow-md">
        <CardHeader>
          <CardTitle>Line Chart</CardTitle>
        </CardHeader>
        <CardContent className="h-40">
          <LineChart />
        </CardContent>
      </Card>

      {/* Bar Chart */}
      <Card className="p-4 shadow-md">
        <CardHeader>
          <CardTitle>Bar Chart</CardTitle>
        </CardHeader>
        <CardContent className="h-40">
          <BarChart />
        </CardContent>
      </Card>

      {/* Pie Chart (Full Width on Small Screens) */}
      <Card className="p-4 shadow-md md:col-span-2">
        <CardHeader>
          <CardTitle>Pie Chart</CardTitle>
        </CardHeader>
        <CardContent className="h-40">
          <PieChart />
        </CardContent>
      </Card>
    </div>
  );
};

export default DashboardCharts;
