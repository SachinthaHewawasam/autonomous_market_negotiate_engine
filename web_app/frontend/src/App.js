import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { authAPI, productsAPI, requestsAPI, negotiationAPI } from './services/api';
import NegotiationViewer from './components/NegotiationViewer';
import WhatIfSimulator from './components/WhatIfSimulator';
import MultiAgentCompetition from './components/MultiAgentCompetition';

function App() {
  const [user, setUser] = useState(null);
  const [loginForm, setLoginForm] = useState({ email: '', password: '' });
  const [registerForm, setRegisterForm] = useState({ email: '', password: '', name: '', role: 'buyer' });
  const [isRegistering, setIsRegistering] = useState(false);
  const [products, setProducts] = useState([]);
  const [requests, setRequests] = useState([]);
  const [activeRequest, setActiveRequest] = useState(null);
  const [viewingRequest, setViewingRequest] = useState(null);
  const [newProduct, setNewProduct] = useState({ name: '', brand: '', quantity: 0, base_price: 0 });
  const [newRequest, setNewRequest] = useState({ product_name: '', brand: '', quantity: 0, max_budget: 0 });
  const [productSuggestions, setProductSuggestions] = useState([]);
  const [brandSuggestions, setBrandSuggestions] = useState([]);
  const [showProductSuggestions, setShowProductSuggestions] = useState(false);
  const [showBrandSuggestions, setShowBrandSuggestions] = useState(false);
  const [showWhatIf, setShowWhatIf] = useState(false);
  const [showMultiAgent, setShowMultiAgent] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem('access_token');
    const savedUser = localStorage.getItem('user');
    if (token && savedUser) {
      setUser(JSON.parse(savedUser));
      loadData();
    }
  }, []);

  const loadData = async () => {
    try {
      const [productsRes, requestsRes] = await Promise.all([
        productsAPI.getAll(),
        requestsAPI.getAll()
      ]);
      setProducts(productsRes.data.products);
      setRequests(requestsRes.data.requests);
    } catch (error) {
      console.error('Error loading data:', error);
    }
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const response = await authAPI.login(loginForm);
      localStorage.setItem('access_token', response.data.access_token);
      localStorage.setItem('user', JSON.stringify(response.data.user));
      setUser(response.data.user);
      loadData();
    } catch (error) {
      alert('Login failed: ' + (error.response?.data?.error || error.message));
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    try {
      const response = await authAPI.register(registerForm);
      localStorage.setItem('access_token', response.data.access_token);
      localStorage.setItem('user', JSON.stringify(response.data.user));
      setUser(response.data.user);
      setIsRegistering(false);
      loadData();
    } catch (error) {
      alert('Registration failed: ' + (error.response?.data?.error || error.message));
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user');
    setUser(null);
    setProducts([]);
    setRequests([]);
  };

  const handleAddProduct = async (e) => {
    e.preventDefault();
    try {
      await productsAPI.create(newProduct);
      setNewProduct({ name: '', brand: '', quantity: 0, base_price: 0 });
      loadData();
      alert('Product added successfully!');
    } catch (error) {
      alert('Error adding product: ' + (error.response?.data?.error || error.message));
    }
  };

  const handleCreateRequest = async (e) => {
    e.preventDefault();
    try {
      await requestsAPI.create(newRequest);
      setNewRequest({ product_name: '', brand: '', quantity: 0, max_budget: 0 });
      loadData();
      alert('Request created successfully!');
    } catch (error) {
      alert('Error creating request: ' + (error.response?.data?.error || error.message));
    }
  };

  const handleProductNameChange = (value) => {
    setNewRequest({ ...newRequest, product_name: value });
    if (value.length > 0) {
      const uniqueProducts = [...new Set(products.map(p => p.name))];
      const filtered = uniqueProducts.filter(name => 
        name.toLowerCase().includes(value.toLowerCase())
      );
      setProductSuggestions(filtered);
      setShowProductSuggestions(true);
    } else {
      setShowProductSuggestions(false);
    }
  };

  const handleBrandChange = (value) => {
    setNewRequest({ ...newRequest, brand: value });
    if (value.length > 0) {
      const uniqueBrands = [...new Set(products.map(p => p.brand))];
      const filtered = uniqueBrands.filter(brand => 
        brand.toLowerCase().includes(value.toLowerCase())
      );
      setBrandSuggestions(filtered);
      setShowBrandSuggestions(true);
    } else {
      setShowBrandSuggestions(false);
    }
  };

  const selectProductSuggestion = (product) => {
    setNewRequest({ ...newRequest, product_name: product });
    setShowProductSuggestions(false);
  };

  const selectBrandSuggestion = (brand) => {
    setNewRequest({ ...newRequest, brand: brand });
    setShowBrandSuggestions(false);
  };

  const handleStartNegotiation = async (requestId) => {
    try {
      await negotiationAPI.start(requestId);
      setActiveRequest(requestId);
    } catch (error) {
      alert('Error starting negotiation: ' + (error.response?.data?.error || error.message));
    }
  };

  if (!user) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-indigo-600 via-purple-600 to-blue-700 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-2xl p-8 max-w-md w-full">
          <div className="text-center mb-8">
            <div className="inline-block p-3 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl mb-4">
              <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
            </div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
              Autonomous Market Simulation
            </h1>
            <p className="text-gray-600">AI-Powered Procurement Platform</p>
          </div>
          
          {!isRegistering ? (
            <form onSubmit={handleLogin} className="space-y-5">
              <h2 className="text-2xl font-bold mb-6 text-gray-800">Welcome Back</h2>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Email Address</label>
                <input
                  type="email"
                  placeholder="you@example.com"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                  value={loginForm.email}
                  onChange={(e) => setLoginForm({ ...loginForm, email: e.target.value })}
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Password</label>
                <input
                  type="password"
                  placeholder="••••••••"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                  value={loginForm.password}
                  onChange={(e) => setLoginForm({ ...loginForm, password: e.target.value })}
                  required
                />
              </div>
              <button
                type="submit"
                className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 transition shadow-lg"
              >
                Sign In
              </button>
              <div className="text-center">
                <button
                  type="button"
                  onClick={() => setIsRegistering(true)}
                  className="text-blue-600 hover:text-blue-700 font-medium"
                >
                  Don't have an account? <span className="underline">Register</span>
                </button>
              </div>
            </form>
          ) : (
            <form onSubmit={handleRegister} className="space-y-5">
              <h2 className="text-2xl font-bold mb-6 text-gray-800">Create Account</h2>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Full Name</label>
                <input
                  type="text"
                  placeholder="John Doe"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                  value={registerForm.name}
                  onChange={(e) => setRegisterForm({ ...registerForm, name: e.target.value })}
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Email Address</label>
                <input
                  type="email"
                  placeholder="you@example.com"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                  value={registerForm.email}
                  onChange={(e) => setRegisterForm({ ...registerForm, email: e.target.value })}
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Password</label>
                <input
                  type="password"
                  placeholder="••••••••"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                  value={registerForm.password}
                  onChange={(e) => setRegisterForm({ ...registerForm, password: e.target.value })}
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Account Type</label>
                <select
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                  value={registerForm.role}
                  onChange={(e) => setRegisterForm({ ...registerForm, role: e.target.value })}
                >
                  <option value="buyer">Buyer - Purchase Products</option>
                  <option value="seller">Seller - Sell Products</option>
                </select>
              </div>
              <button
                type="submit"
                className="w-full bg-gradient-to-r from-green-600 to-teal-600 text-white py-3 rounded-lg font-semibold hover:from-green-700 hover:to-teal-700 transition shadow-lg"
              >
                Create Account
              </button>
              <div className="text-center">
                <button
                  type="button"
                  onClick={() => setIsRegistering(false)}
                  className="text-blue-600 hover:text-blue-700 font-medium"
                >
                  Already have an account? <span className="underline">Login</span>
                </button>
              </div>
            </form>
          )}
        </div>
      </div>
    );
  }

  if (viewingRequest) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-6xl mx-auto">
          <div className="bg-white rounded-lg shadow p-4 mb-6 flex justify-between items-center">
            <h1 className="text-2xl font-bold">Request Details</h1>
            <button
              onClick={() => { setViewingRequest(null); loadData(); }}
              className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
            >
              Back to Dashboard
            </button>
          </div>
          <NegotiationViewer
            requestId={viewingRequest}
            onComplete={() => { setViewingRequest(null); loadData(); }}
            viewOnly={true}
          />
        </div>
      </div>
    );
  }

  if (activeRequest) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-6xl mx-auto">
          <div className="bg-white rounded-lg shadow p-4 mb-6 flex justify-between items-center">
            <h1 className="text-2xl font-bold">Negotiation in Progress</h1>
            <button
              onClick={() => { setActiveRequest(null); loadData(); }}
              className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
            >
              Back to Dashboard
            </button>
          </div>
          <NegotiationViewer
            requestId={activeRequest}
            onComplete={() => { setActiveRequest(null); loadData(); }}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <nav className="bg-white shadow-lg border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-4">
              <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl">
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Market Simulation
                </h1>
                <p className="text-sm text-gray-600">{user.role === 'buyer' ? 'Buyer' : 'Seller'} Dashboard</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-right">
                <p className="text-sm text-gray-500">Welcome back</p>
                <p className="font-semibold text-gray-800">{user.name}</p>
              </div>
              <button
                onClick={handleLogout}
                className="px-4 py-2 bg-gradient-to-r from-red-500 to-red-600 text-white rounded-lg hover:from-red-600 hover:to-red-700 transition shadow-md font-medium"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto p-6">
        {user.role === 'seller' ? (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-lg p-8 border border-gray-200">
              <div className="flex items-center space-x-3 mb-6">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                  </svg>
                </div>
                <h2 className="text-2xl font-bold text-gray-800">Add New Product</h2>
              </div>
              <form onSubmit={handleAddProduct} className="grid grid-cols-2 gap-6">
                <input
                  type="text"
                  placeholder="Product Name"
                  className="px-4 py-2 border rounded"
                  value={newProduct.name}
                  onChange={(e) => setNewProduct({ ...newProduct, name: e.target.value })}
                  required
                />
                <input
                  type="text"
                  placeholder="Brand"
                  className="px-4 py-2 border rounded"
                  value={newProduct.brand}
                  onChange={(e) => setNewProduct({ ...newProduct, brand: e.target.value })}
                  required
                />
                <input
                  type="number"
                  placeholder="Quantity"
                  className="px-4 py-2 border rounded"
                  value={newProduct.quantity}
                  onChange={(e) => setNewProduct({ ...newProduct, quantity: parseInt(e.target.value) })}
                  required
                />
                <input
                  type="number"
                  step="0.01"
                  placeholder="Base Price"
                  className="px-4 py-2 border rounded"
                  value={newProduct.base_price}
                  onChange={(e) => setNewProduct({ ...newProduct, base_price: parseFloat(e.target.value) })}
                  required
                />
                <button
                  type="submit"
                  className="col-span-2 bg-blue-600 text-white py-2 rounded hover:bg-blue-700"
                >
                  Add Product
                </button>
              </form>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-bold mb-4">My Products</h2>
              <div className="overflow-x-auto">
                <table className="min-w-full">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-2 text-left">Name</th>
                      <th className="px-4 py-2 text-left">Brand</th>
                      <th className="px-4 py-2 text-left">Quantity</th>
                      <th className="px-4 py-2 text-left">Price</th>
                    </tr>
                  </thead>
                  <tbody>
                    {products.filter(p => p.seller_id === user.id).map((product) => (
                      <tr key={product.id} className="border-t">
                        <td className="px-4 py-2">{product.name}</td>
                        <td className="px-4 py-2">{product.brand}</td>
                        <td className="px-4 py-2">{product.quantity}</td>
                        <td className="px-4 py-2">${product.base_price.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-lg p-8 border border-gray-200">
              <div className="flex items-center space-x-3 mb-6">
                <div className="p-2 bg-green-100 rounded-lg">
                  <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                  </svg>
                </div>
                <h2 className="text-2xl font-bold text-gray-800">Create Procurement Request</h2>
              </div>
              <form onSubmit={handleCreateRequest} className="grid grid-cols-2 gap-6">
                <div className="relative">
                  <label className="block text-sm font-medium text-gray-700 mb-2">Product Name</label>
                  <input
                    type="text"
                    placeholder="e.g., Biscuits"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                    value={newRequest.product_name}
                    onChange={(e) => handleProductNameChange(e.target.value)}
                    onBlur={() => setTimeout(() => setShowProductSuggestions(false), 200)}
                    required
                  />
                  {showProductSuggestions && productSuggestions.length > 0 && (
                    <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-48 overflow-y-auto">
                      {productSuggestions.map((product, idx) => (
                        <div
                          key={idx}
                          className="px-4 py-2 hover:bg-blue-50 cursor-pointer transition"
                          onClick={() => selectProductSuggestion(product)}
                        >
                          {product}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                <div className="relative">
                  <label className="block text-sm font-medium text-gray-700 mb-2">Brand</label>
                  <input
                    type="text"
                    placeholder="e.g., Brand X"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                    value={newRequest.brand}
                    onChange={(e) => handleBrandChange(e.target.value)}
                    onBlur={() => setTimeout(() => setShowBrandSuggestions(false), 200)}
                    required
                  />
                  {showBrandSuggestions && brandSuggestions.length > 0 && (
                    <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-48 overflow-y-auto">
                      {brandSuggestions.map((brand, idx) => (
                        <div
                          key={idx}
                          className="px-4 py-2 hover:bg-blue-50 cursor-pointer transition"
                          onClick={() => selectBrandSuggestion(brand)}
                        >
                          {brand}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Quantity (units)</label>
                  <input
                    type="number"
                    placeholder="e.g., 120"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                    value={newRequest.quantity}
                    onChange={(e) => setNewRequest({ ...newRequest, quantity: parseInt(e.target.value) })}
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Maximum Budget ($)</label>
                  <input
                    type="number"
                    step="0.01"
                    placeholder="e.g., 1000.00"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                    value={newRequest.max_budget}
                    onChange={(e) => setNewRequest({ ...newRequest, max_budget: parseFloat(e.target.value) })}
                    required
                  />
                </div>
                <button
                  type="submit"
                  className="col-span-2 bg-gradient-to-r from-green-600 to-teal-600 text-white py-3 rounded-lg font-semibold hover:from-green-700 hover:to-teal-700 transition shadow-lg"
                >
                  Create Request
                </button>
              </form>
            </div>

            {/* What-If Simulator Toggle */}
            <div className="flex justify-center">
              <button
                onClick={() => setShowWhatIf(!showWhatIf)}
                className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg font-semibold hover:from-purple-700 hover:to-pink-700 transition shadow-lg flex items-center space-x-2"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
                <span>{showWhatIf ? 'Hide' : 'Show'} What-If Simulator</span>
              </button>
            </div>

            {/* What-If Simulator */}
            {showWhatIf && (
              <WhatIfSimulator 
                products={products}
                initialRequest={newRequest}
              />
            )}

            {/* Multi-Agent Competition Toggle */}
            <div className="flex justify-center">
              <button
                onClick={() => setShowMultiAgent(!showMultiAgent)}
                className="px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg font-semibold hover:from-indigo-700 hover:to-purple-700 transition shadow-lg flex items-center space-x-2"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                </svg>
                <span>{showMultiAgent ? 'Hide' : 'Show'} Multi-Agent Competition</span>
              </button>
            </div>

            {/* Multi-Agent Competition */}
            {showMultiAgent && <MultiAgentCompetition />}

            <div className="bg-white rounded-xl shadow-lg p-8 border border-gray-200">
              <div className="flex items-center space-x-3 mb-6">
                <div className="p-2 bg-purple-100 rounded-lg">
                  <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                  </svg>
                </div>
                <h2 className="text-2xl font-bold text-gray-800">Available Products</h2>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {products.map((product) => (
                  <div key={product.id} className="border border-gray-200 rounded-xl p-6 hover:shadow-xl transition-all duration-300 hover:border-blue-300 bg-gradient-to-br from-white to-gray-50">
                    <div className="flex justify-between items-start mb-3">
                      <div>
                        <h3 className="font-bold text-lg text-gray-800">{product.name}</h3>
                        <p className="text-sm text-gray-500">{product.brand}</p>
                      </div>
                      <span className="px-3 py-1 bg-green-100 text-green-700 text-xs font-semibold rounded-full">
                        {(product.trust_score * 100).toFixed(0)}% Trust
                      </span>
                    </div>
                    <div className="space-y-2 mb-4">
                      <div className="flex items-center text-sm text-gray-600">
                        <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                        </svg>
                        {product.seller_name}
                      </div>
                      <div className="flex items-center text-sm text-gray-600">
                        <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                        </svg>
                        {product.quantity} units available
                      </div>
                    </div>
                    <div className="pt-4 border-t border-gray-200">
                      <p className="text-2xl font-bold text-green-600">${product.base_price.toFixed(2)}<span className="text-sm text-gray-500">/unit</span></p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-8 border border-gray-200">
              <div className="flex items-center space-x-3 mb-6">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                  </svg>
                </div>
                <h2 className="text-2xl font-bold text-gray-800">My Requests</h2>
              </div>
              <div className="space-y-4">
                {requests.length === 0 ? (
                  <div className="text-center py-12 text-gray-500">
                    <svg className="w-16 h-16 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <p className="text-lg font-medium">No requests yet</p>
                    <p className="text-sm">Create your first procurement request above</p>
                  </div>
                ) : (
                  requests.map((req) => (
                    <div 
                      key={req.id} 
                      className="border border-gray-200 rounded-xl p-6 hover:shadow-lg transition-all bg-gradient-to-r from-white to-gray-50 cursor-pointer"
                      onClick={() => {
                        if (req.status !== 'pending') {
                          setViewingRequest(req.id);
                        }
                      }}
                    >
                      <div className="flex justify-between items-start">
                        <div className="flex-1">
                          <div className="flex items-center space-x-3 mb-3">
                            <h3 className="font-bold text-lg text-gray-800">{req.product_name}</h3>
                            <span className="text-gray-400">•</span>
                            <span className="text-gray-600">{req.brand}</span>
                          </div>
                          <div className="grid grid-cols-2 gap-4 mb-3">
                            <div className="flex items-center text-sm text-gray-600">
                              <svg className="w-4 h-4 mr-2 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                              </svg>
                              <span className="font-medium">{req.quantity} units</span>
                            </div>
                            <div className="flex items-center text-sm text-gray-600">
                              <svg className="w-4 h-4 mr-2 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                              </svg>
                              <span className="font-medium">${req.max_budget.toFixed(2)}</span>
                            </div>
                          </div>
                          <span className={`inline-flex items-center px-3 py-1 text-xs font-semibold rounded-full ${
                            req.status === 'completed' ? 'bg-green-100 text-green-800' :
                            req.status === 'pending' ? 'bg-yellow-100 text-yellow-800' :
                            req.status === 'negotiating' ? 'bg-blue-100 text-blue-800' :
                            req.status === 'pending_approval' ? 'bg-purple-100 text-purple-800' :
                            'bg-red-100 text-red-800'
                          }`}>
                            <span className={`w-2 h-2 rounded-full mr-2 ${
                              req.status === 'completed' ? 'bg-green-500' :
                              req.status === 'pending' ? 'bg-yellow-500' :
                              req.status === 'negotiating' ? 'bg-blue-500 animate-pulse' :
                              req.status === 'pending_approval' ? 'bg-purple-500 animate-pulse' :
                              'bg-red-500'
                            }`}></span>
                            {req.status.replace('_', ' ').toUpperCase()}
                          </span>
                        </div>
                        {req.status === 'pending' ? (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleStartNegotiation(req.id);
                            }}
                            className="ml-4 px-6 py-3 bg-gradient-to-r from-green-600 to-teal-600 text-white rounded-lg font-semibold hover:from-green-700 hover:to-teal-700 transition shadow-lg flex items-center space-x-2"
                          >
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                            </svg>
                            <span>Start Negotiation</span>
                          </button>
                        ) : (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setViewingRequest(req.id);
                            }}
                            className="ml-4 px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg font-semibold hover:from-blue-700 hover:to-indigo-700 transition shadow-lg flex items-center space-x-2"
                          >
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                            </svg>
                            <span>View Details</span>
                          </button>
                        )}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
