import React from "react";

const AppNavBar: React.FC = () => {
  return (
    <nav className="flex justify-between items-center px-8 py-4 bg-white border-b">
      {/* Logo */}
      <div className="text-lg font-bold">Logo</div>

      {/* Navigation Links */}
      <ul className="hidden md:flex gap-6 text-gray-700">
        <li><a href="#" className="hover:text-black">Link One</a></li>
        <li><a href="#" className="hover:text-black">Link Two</a></li>
        <li><a href="#" className="hover:text-black">Link Three</a></li>
        <li>
          <a href="#" className="hover:text-black flex items-center">
            Link Four ▼
          </a>
        </li>
      </ul>

      {/* Search Bar (Optional) */}
      <div className="hidden md:flex border rounded px-3 py-1 text-gray-600">
        🔍 <input type="text" placeholder="Search" className="ml-2 outline-none"/>
      </div>
    </nav>
  );
};

export default AppNavBar;
