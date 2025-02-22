import React from "react";
import AppNavBar from "./appNavBar";

type AppShellProps = {
  children: React.ReactNode;
};

const AppShell: React.FC<AppShellProps> = ({ children }) => {
  return (
    <div className="flex flex-col min-h-screen">
      <AppNavBar />
      <main className="flex-1 container mx-auto p-8">{children}</main>

      {/* Footer */}
      <footer className="mt-auto bg-white border-t p-6 text-center text-gray-600">
        <div className="flex justify-between items-center max-w-4xl mx-auto">
          {/* Footer Logo */}
          <div className="text-lg font-bold">Logo</div>

          {/* Footer Links */}
          <ul className="hidden md:flex gap-6">
            <li><a href="#" className="hover:text-black">Link One</a></li>
            <li><a href="#" className="hover:text-black">Link Two</a></li>
            <li><a href="#" className="hover:text-black">Link Three</a></li>
            <li><a href="#" className="hover:text-black">Link Four</a></li>
            <li><a href="#" className="hover:text-black">Link Five</a></li>
          </ul>

          {/* Social Icons */}
          <div className="flex gap-4">
            <a href="#" className="hover:text-black">🌐</a>
            <a href="#" className="hover:text-black">📘</a>
            <a href="#" className="hover:text-black">🐦</a>
            <a href="#" className="hover:text-black">📸</a>
          </div>
        </div>

        <div className="mt-4 text-sm">
          © 2023 Relume. All rights reserved. 
          <a href="#" className="ml-4 hover:underline">Privacy Policy</a> | 
          <a href="#" className="hover:underline"> Terms of Service</a> | 
          <a href="#" className="hover:underline"> Cookies Settings</a>
        </div>
      </footer>
    </div>
  );
};

export default AppShell;
