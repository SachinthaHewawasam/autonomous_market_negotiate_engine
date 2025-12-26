/**
 * Real-time negotiation visualization component
 */
import React, { useState, useEffect } from 'react';
import { io } from 'socket.io-client';
import { negotiationAPI } from '../services/api';

const SOCKET_URL = process.env.REACT_APP_SOCKET_URL || 'http://localhost:5000';

export default function NegotiationViewer({ requestId, onComplete, viewOnly = false }) {
  const [socket, setSocket] = useState(null);
  const [marketState, setMarketState] = useState(null);
  const [steps, setSteps] = useState([]);
  const [deal, setDeal] = useState(null);
  const [status, setStatus] = useState('connecting');
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    // If view only, load existing data
    if (viewOnly) {
      loadNegotiationHistory();
      return;
    }

    // Connect to WebSocket
    const newSocket = io(SOCKET_URL);
    setSocket(newSocket);

    newSocket.on('connect', () => {
      console.log('Connected to negotiation server');
      setStatus('connected');
      setLogs(prev => [...prev, { time: new Date(), type: 'info', message: '✓ Connected to negotiation server' }]);
      newSocket.emit('join_request', { request_id: requestId });
    });

    newSocket.on('negotiation_started', (data) => {
      console.log('Negotiation started:', data);
      setStatus('negotiating');
      setLogs(prev => [...prev, { 
        time: new Date(), 
        type: 'success', 
        message: '🚀 Negotiation started - RL Agent initializing...' 
      }]);
    });

    newSocket.on('market_state', (data) => {
      console.log('Market state:', data);
      setMarketState(data);
      setLogs(prev => [...prev, { 
        time: new Date(), 
        type: 'info', 
        message: `📊 Market analyzed: ${data.sellers.length} sellers found, requesting ${data.requested_quantity} units with $${data.max_budget.toFixed(2)} budget` 
      }]);
    });

    newSocket.on('negotiation_step', (data) => {
      console.log('Negotiation step:', data);
      setSteps((prev) => [...prev, data.step]);
      const step = data.step;
      setLogs(prev => [...prev, { 
        time: new Date(), 
        type: step.reward > 0 ? 'success' : 'warning', 
        message: `🤖 Round ${step.round_number}: ${step.action_type} - Reward: ${step.reward > 0 ? '+' : ''}${step.reward.toFixed(2)}` 
      }]);
    });

    newSocket.on('deal_ready', (data) => {
      console.log('Deal ready:', data);
      setDeal(data.deal);
      setStatus('pending_approval');
      setLogs(prev => [...prev, { 
        time: new Date(), 
        type: 'success', 
        message: `✅ Deal negotiated! Total: $${data.deal.total_cost.toFixed(2)}, Savings: $${data.deal.savings.toFixed(2)}` 
      }]);
    });

    newSocket.on('negotiation_failed', (data) => {
      console.log('Negotiation failed:', data);
      setStatus('failed');
      setLogs(prev => [...prev, { 
        time: new Date(), 
        type: 'error', 
        message: `❌ Negotiation failed: ${data.reason}` 
      }]);
    });

    return () => {
      newSocket.emit('leave_request', { request_id: requestId });
      newSocket.disconnect();
    };
  }, [requestId, viewOnly]);

  const loadNegotiationHistory = async () => {
    try {
      const response = await negotiationAPI.getById(requestId);
      const req = response.data.request;
      
      if (req.negotiation) {
        setStatus(req.status);
        setSteps(req.negotiation.steps || []);
        setDeal(req.negotiation.deal || null);
        
        // Simulate logs from steps
        const historyLogs = [
          { time: new Date(req.created_at), type: 'info', message: '✓ Negotiation completed' }
        ];
        
        if (req.negotiation.steps) {
          req.negotiation.steps.forEach(step => {
            historyLogs.push({
              time: new Date(step.timestamp),
              type: step.reward > 0 ? 'success' : 'warning',
              message: `🤖 Round ${step.round_number}: ${step.action_type} - Reward: ${step.reward > 0 ? '+' : ''}${step.reward.toFixed(2)}`
            });
          });
        }
        
        setLogs(historyLogs);
      }
    } catch (error) {
      console.error('Error loading negotiation history:', error);
    }
  };

  const handleApprove = async () => {
    try {
      await negotiationAPI.approveDeal(deal.id);
      setStatus('approved');
      if (onComplete) onComplete('approved');
    } catch (error) {
      console.error('Error approving deal:', error);
    }
  };

  const handleReject = async () => {
    try {
      await negotiationAPI.rejectDeal(deal.id);
      setStatus('rejected');
      if (onComplete) onComplete('rejected');
    } catch (error) {
      console.error('Error rejecting deal:', error);
    }
  };

  return (
    <div className="space-y-6">
      {/* Market State */}
      {marketState && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Market Overview</h3>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <p className="text-sm text-gray-600">Requested Quantity</p>
              <p className="text-2xl font-bold">{marketState.requested_quantity} units</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Maximum Budget</p>
              <p className="text-2xl font-bold">${marketState.max_budget.toFixed(2)}</p>
            </div>
          </div>
          
          <h4 className="font-semibold mb-2">Available Sellers</h4>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Seller</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Stock</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Price/unit</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Trust</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {marketState.sellers.map((seller) => (
                  <tr key={seller.id}>
                    <td className="px-4 py-2 whitespace-nowrap">{seller.seller_name}</td>
                    <td className="px-4 py-2 whitespace-nowrap">{seller.stock} units</td>
                    <td className="px-4 py-2 whitespace-nowrap">${seller.price.toFixed(2)}</td>
                    <td className="px-4 py-2 whitespace-nowrap">
                      <span className="px-2 py-1 text-xs rounded-full bg-green-100 text-green-800">
                        {(seller.trust * 100).toFixed(0)}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Live Activity Log */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          <span className="w-3 h-3 bg-green-500 rounded-full mr-2 animate-pulse"></span>
          Live Activity Log
        </h3>
        <div className="bg-gray-900 rounded-lg p-4 h-64 overflow-y-auto font-mono text-sm">
          {logs.map((log, idx) => (
            <div key={idx} className={`mb-2 ${
              log.type === 'success' ? 'text-green-400' :
              log.type === 'error' ? 'text-red-400' :
              log.type === 'warning' ? 'text-yellow-400' :
              'text-blue-400'
            }`}>
              <span className="text-gray-500">[{log.time.toLocaleTimeString()}]</span> {log.message}
            </div>
          ))}
          {logs.length === 0 && (
            <div className="text-gray-500 text-center py-8">Waiting for negotiation to start...</div>
          )}
        </div>
      </div>

      {/* Negotiation Steps */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Negotiation Process</h3>
        
        {status === 'connecting' && (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
            <p className="mt-4 text-gray-600">Connecting to negotiation server...</p>
          </div>
        )}

        {status === 'negotiating' && steps.length === 0 && (
          <div className="text-center py-8">
            <div className="relative">
              <div className="flex justify-center space-x-2 mb-6">
                <div className="w-3 h-3 bg-blue-600 rounded-full animate-bounce" style={{animationDelay: '0ms'}}></div>
                <div className="w-3 h-3 bg-blue-600 rounded-full animate-bounce" style={{animationDelay: '150ms'}}></div>
                <div className="w-3 h-3 bg-blue-600 rounded-full animate-bounce" style={{animationDelay: '300ms'}}></div>
              </div>
              <div className="space-y-3">
                <div className="flex items-center justify-center space-x-3">
                  <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                  <p className="text-lg font-semibold text-gray-700">RL Agent analyzing market...</p>
                </div>
                <div className="bg-blue-50 rounded-lg p-4 max-w-md mx-auto">
                  <p className="text-sm text-blue-800">
                    <span className="font-semibold">Current Step:</span> Evaluating seller options and forming optimal coalition
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="space-y-4">
          {steps.map((step, index) => (
            <div key={index} className="border-l-4 border-blue-500 pl-4 py-2">
              <div className="flex items-center justify-between mb-2">
                <span className="font-semibold text-blue-600">
                  Round {step.round_number}: {step.action_type}
                </span>
                <span className={`px-2 py-1 text-xs rounded ${
                  step.reward > 50 ? 'bg-green-100 text-green-800' :
                  step.reward > 0 ? 'bg-yellow-100 text-yellow-800' :
                  'bg-red-100 text-red-800'
                }`}>
                  Reward: {step.reward > 0 ? '+' : ''}{step.reward.toFixed(2)}
                </span>
              </div>
              {step.explanation && (
                <p className="text-sm text-gray-600 whitespace-pre-line">{step.explanation}</p>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Deal Approval */}
      {deal && status === 'pending_approval' && (
        <div className="bg-yellow-50 border-2 border-yellow-400 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4 text-yellow-900">
            🎉 Deal Ready for Your Approval
          </h3>
          
          <div className="bg-white rounded p-4 mb-4">
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <p className="text-sm text-gray-600">Total Cost</p>
                <p className="text-2xl font-bold text-green-600">${deal.total_cost.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Savings</p>
                <p className="text-2xl font-bold text-blue-600">${deal.savings.toFixed(2)}</p>
              </div>
            </div>

            {deal.sellers && deal.sellers.length > 0 && (
              <div>
                <h4 className="font-semibold mb-2">Selected Sellers</h4>
                <ul className="space-y-2">
                  {deal.sellers.map((seller, idx) => (
                    <li key={idx} className="flex justify-between items-center bg-gray-50 p-2 rounded">
                      <span>{seller.seller_name}</span>
                      <span className="text-sm text-gray-600">
                        {seller.quantity} units @ ${seller.price_per_unit.toFixed(2)} = ${seller.subtotal.toFixed(2)}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          <div className="flex space-x-4">
            <button
              onClick={handleApprove}
              className="flex-1 bg-green-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-green-700 transition"
            >
              ✓ Approve Deal
            </button>
            <button
              onClick={handleReject}
              className="flex-1 bg-red-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-red-700 transition"
            >
              ✗ Reject Deal
            </button>
          </div>
        </div>
      )}

      {/* Final Status */}
      {status === 'approved' && (
        <div className="bg-green-50 border-2 border-green-400 rounded-lg p-6 text-center">
          <h3 className="text-xl font-bold text-green-900 mb-2">✅ Deal Approved!</h3>
          <p className="text-green-700">The procurement order will be processed.</p>
        </div>
      )}

      {status === 'rejected' && (
        <div className="bg-red-50 border-2 border-red-400 rounded-lg p-6 text-center">
          <h3 className="text-xl font-bold text-red-900 mb-2">❌ Deal Rejected</h3>
          <p className="text-red-700">You can create a new request with adjusted parameters.</p>
        </div>
      )}

      {status === 'failed' && (
        <div className="bg-red-50 border-2 border-red-400 rounded-lg p-6 text-center">
          <h3 className="text-xl font-bold text-red-900 mb-2">❌ Negotiation Failed</h3>
          <p className="text-red-700">Unable to find a suitable deal within constraints.</p>
        </div>
      )}
    </div>
  );
}
