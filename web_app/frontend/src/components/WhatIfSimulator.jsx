import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

export default function WhatIfSimulator({ products, initialRequest }) {
  const [params, setParams] = useState({
    product_name: initialRequest?.product_name || '',
    brand: initialRequest?.brand || '',
    quantity: initialRequest?.quantity || 100,
    max_budget: initialRequest?.max_budget || 1000
  });
  
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [competitionResult, setCompetitionResult] = useState(null);
  const [simulatingCompetition, setSimulatingCompetition] = useState(false);

  const runPrediction = async () => {
    if (!params.product_name || !params.brand) return;
    
    setLoading(true);
    try {
      const token = localStorage.getItem('access_token');
      const response = await axios.post(`${API_URL}/what-if`, params, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setPrediction(response.data);
    } catch (error) {
      console.error('Prediction error:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (params.product_name && params.brand) {
      const timer = setTimeout(() => runPrediction(), 500);
      return () => clearTimeout(timer);
    }
  }, [params]);

  return (
    <div className="bg-white rounded-xl shadow-lg p-8 border border-gray-200">
      <div className="flex items-center space-x-3 mb-6">
        <div className="p-2 bg-purple-100 rounded-lg">
          <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
        </div>
        <h2 className="text-2xl font-bold text-gray-800">What-If Simulator</h2>
      </div>

      {!params.product_name || !params.brand ? (
        <div className="text-center py-12 bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg border-2 border-dashed border-purple-300">
          <svg className="w-16 h-16 mx-auto mb-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p className="text-lg font-semibold text-gray-700 mb-2">Fill in Product Details First</p>
          <p className="text-gray-600">Enter Product Name and Brand in the form above to start simulating scenarios</p>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-6 mb-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Quantity (units)</label>
              <input
                type="range"
                min="20"
                max="200"
                value={params.quantity}
                onChange={(e) => setParams({...params, quantity: parseInt(e.target.value)})}
                className="w-full"
              />
              <div className="text-center font-bold text-lg text-blue-600">{params.quantity} units</div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Maximum Budget ($)</label>
              <input
                type="range"
                min="500"
                max="2000"
                step="50"
                value={params.max_budget}
                onChange={(e) => setParams({...params, max_budget: parseInt(e.target.value)})}
                className="w-full"
              />
              <div className="text-center font-bold text-lg text-green-600">${params.max_budget}</div>
            </div>
          </div>
        </>
      )}

      {(params.product_name && params.brand) && (
        <>
          {loading && (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600 mx-auto"></div>
              <p className="mt-4 text-gray-600">Analyzing scenario...</p>
            </div>
          )}

          {prediction && !loading && (
        <div className="space-y-6">
          {/* Success Probability */}
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-800">Predicted Outcome</h3>
              <span className={`px-4 py-2 rounded-full font-bold ${
                prediction.success_probability > 0.8 ? 'bg-green-100 text-green-800' :
                prediction.success_probability > 0.5 ? 'bg-yellow-100 text-yellow-800' :
                'bg-red-100 text-red-800'
              }`}>
                {(prediction.success_probability * 100).toFixed(0)}% Success
              </span>
            </div>

            <div className="grid grid-cols-3 gap-4">
              <div className="bg-white rounded-lg p-4">
                <div className="text-sm text-gray-600 mb-1">Estimated Cost</div>
                <div className="text-2xl font-bold text-gray-800">
                  ${prediction.estimated_cost?.most_likely?.toFixed(2) || 'N/A'}
                </div>
                <div className="text-xs text-gray-500">
                  ${prediction.estimated_cost?.min?.toFixed(2)} - ${prediction.estimated_cost?.max?.toFixed(2)}
                </div>
              </div>

              <div className="bg-white rounded-lg p-4">
                <div className="text-sm text-gray-600 mb-1">Potential Savings</div>
                <div className="text-2xl font-bold text-green-600">
                  ${prediction.predicted_savings?.toFixed(2) || '0.00'}
                </div>
                <div className="text-xs text-gray-500">
                  {((prediction.predicted_savings / params.max_budget) * 100).toFixed(1)}% of budget
                </div>
              </div>

              <div className="bg-white rounded-lg p-4">
                <div className="text-sm text-gray-600 mb-1">Risk Level</div>
                <div className={`text-2xl font-bold ${
                  prediction.risk_level === 'Very Low' || prediction.risk_level === 'Low' ? 'text-green-600' :
                  prediction.risk_level === 'Medium' ? 'text-yellow-600' :
                  'text-red-600'
                }`}>
                  {prediction.risk_level}
                </div>
                <div className="text-xs text-gray-500">
                  {prediction.delivery_estimate || 'N/A'} delivery
                </div>
              </div>
            </div>
          </div>

          {/* Strategy */}
          <div className="bg-white border-2 border-purple-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Recommended Strategy</h3>
            <div className="flex items-center space-x-3 mb-4">
              <span className="px-4 py-2 bg-purple-100 text-purple-800 rounded-lg font-semibold">
                {prediction.predicted_action}
              </span>
              {prediction.coalition_needed && (
                <span className="text-sm text-gray-600">
                  {prediction.num_sellers_needed} sellers needed
                </span>
              )}
            </div>

            {prediction.recommended_sellers && prediction.recommended_sellers.length > 0 && (
              <div className="space-y-2">
                <div className="text-sm font-medium text-gray-700 mb-2">Recommended Sellers:</div>
                {prediction.recommended_sellers.map((seller, idx) => (
                  <div key={idx} className="flex items-center justify-between bg-gray-50 rounded-lg p-3">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center text-purple-600 font-bold">
                        {idx + 1}
                      </div>
                      <div>
                        <div className="font-semibold text-gray-800">{seller.name}</div>
                        <div className="text-sm text-gray-600">
                          {seller.quantity} units @ ${seller.price.toFixed(2)}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-bold text-gray-800">
                        ${(seller.quantity * seller.price).toFixed(2)}
                      </div>
                      <div className="text-xs text-gray-500">
                        {(seller.trust * 100).toFixed(0)}% trust
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Insights */}
          <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
            <div className="flex items-start space-x-3">
              <svg className="w-6 h-6 text-blue-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div className="text-sm text-blue-800">
                {prediction.success_probability > 0.8 ? (
                  <p><strong>Great scenario!</strong> High probability of success with good savings potential.</p>
                ) : prediction.success_probability > 0.5 ? (
                  <p><strong>Moderate scenario.</strong> Consider adjusting budget or quantity for better outcomes.</p>
                ) : (
                  <p><strong>Challenging scenario.</strong> Try increasing budget or reducing quantity.</p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
        </>
      )}
    </div>
  );
}
