import React, { ReactNode } from 'react';

interface DashboardLayoutProps {
  children?: ReactNode;
  sidebarContent?: ReactNode;
  mainContent?: ReactNode;
}

const DashboardLayout: React.FC<DashboardLayoutProps> = ({
  children,
  sidebarContent,
  mainContent
}) => {
  return (
    <main className="w-full h-screen flex bg-white">
      <div className="w-1/6 h-full border-r border-solid border-black bg-opacity-90">
        {sidebarContent}
      </div>
      <div className="w-5/6 h-full">
        {mainContent || children}
      </div>
    </main>
  );
};

export default DashboardLayout;