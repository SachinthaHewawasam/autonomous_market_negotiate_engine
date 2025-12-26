import React, { useState } from 'react';
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

export default function MultiAgentCompetition() {
  const [numAgents, setNumAgents] = useState(3);
  const [agentRequests, setAgentRequests] = useState([
    { buyer_name: 'Agent Alpha', product_name: 'Biscuits', brand: 'Brand X', quantity: 40, max_budget: 400 },
    { buyer_name: 'Agent Beta', product_name: 'Biscuits', brand: 'Brand X', quantity: 45, max_budget: 450 },
    { buyer_name: 'Agent Gamma', product_name: 'Biscuits', brand: 'Brand X', quantity: 35, max_budget: 380 },
  ]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [currentRound, setCurrentRound] = useState(0);

  const runCompetition = async () => {
    setLoading(true);
    setResults(null);
    setCurrentRound(0);

    try {
      const token = localStorage.getItem('access_token');
      const response = await axios.post(
        `${API_URL}/multi-agent/compete`,
        { requests: agentRequests },
        { headers: { Authorization: `Bearer ${token}` } }
      );

      setResults(response.data);
      
      // Animate through rounds
      if (response.data.rounds) {
        for (let i = 0; i <= response.data.rounds.length; i++) {
          await new Promise(resolve => setTimeout(resolve, 1000));
          setCurrentRound(i);
        }
      }
    } catch (error) {
      console.error('Competition error:', error);
      alert('Error running competition: ' + (error.response?.data?.error || error.message));
    } finally {
      setLoading(false);
    }
  };

  const runDemo = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_URL}/multi-agent/demo`);
      setResults(response.data);
    } catch (error) {
      console.error('Demo error:', error);
    } finally {
      setLoading(false);
    }
  };

  const updateAgentRequest = (index, field, value) => {
    const updated = [...agentRequests];
    updated[index][field] = value;
    setAgentRequests(updated);
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-8 border border-gray-200">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-gradient-to-r from-purple-100 to-pink-100 rounded-lg">
            <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-800">Multi-Agent Competition</h2>
            <p className="text-sm text-gray-600">Watch RL agents compete for limited resources</p>
          </div>
        </div>
        <button
          onClick={runDemo}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition text-sm"
        >
          Run Quick Demo
        </button>
      </div>

      {/* Agent Configuration */}
      <div className="space-y-4 mb-6">
        <h3 className="font-semibold text-gray-700">Configure Competing Agents</h3>
        {agentRequests.map((agent, index) => (
          <div key={index} className="grid grid-cols-5 gap-3 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Agent Name</label>
              <input
                type="text"
                value={agent.buyer_name}
                onChange={(e) => updateAgentRequest(index, 'buyer_name', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Product</label>
              <input
                type="text"
                value={agent.product_name}
                onChange={(e) => updateAgentRequest(index, 'product_name', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Brand</label>
              <input
                type="text"
                value={agent.brand}
                onChange={(e) => updateAgentRequest(index, 'brand', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Quantity</label>
              <input
                type="number"
                value={agent.quantity}
                onChange={(e) => updateAgentRequest(index, 'quantity', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Budget ($)</label>
              <input
                type="number"
                value={agent.max_budget}
                onChange={(e) => updateAgentRequest(index, 'max_budget', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
              />
            </div>
          </div>
        ))}
      </div>

      {/* Start Competition Button */}
      <button
        onClick={runCompetition}
        disabled={loading}
        className="w-full py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg font-semibold hover:from-purple-700 hover:to-pink-700 transition shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? '🏁 Competition Running...' : '🚀 Start Competition'}
      </button>

      {/* Results */}
      {results && (
        <div className="mt-8 space-y-6">
          {/* Winner Announcement */}
          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 border-2 border-yellow-400 rounded-xl p-6">
            <div className="flex items-center space-x-3 mb-4">
              <svg className="w-8 h-8 text-yellow-600" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
              </svg>
              <div>
                <h3 className="text-2xl font-bold text-gray-800">🏆 Winner: {results.winner}</h3>
                <p className="text-gray-600">Total Reward: {results.winner_reward?.toFixed(2)}</p>
              </div>
            </div>
            <p className="text-sm text-gray-700">Completed in {results.num_rounds} rounds</p>
          </div>

          {/* Agent Performance Metrics */}
          <div className="grid grid-cols-3 gap-4">
            {Object.entries(results.buyer_metrics || {}).map(([name, metrics]) => (
              <div key={name} className={`p-4 rounded-lg border-2 ${
                name === results.winner 
                  ? 'bg-green-50 border-green-400' 
                  : 'bg-gray-50 border-gray-300'
              }`}>
                <h4 className="font-bold text-gray-800 mb-2">{name}</h4>
                <div className="space-y-1 text-sm">
                  <p className="text-gray-700">
                    <span className="font-medium">Reward:</span> {metrics.total_reward?.toFixed(2)}
                  </p>
                  <p className="text-gray-700">
                    <span className="font-medium">Rounds:</span> {metrics.rounds_active}
                  </p>
                  <p className="text-gray-700">
                    <span className="font-medium">Conflicts:</span> {metrics.conflicts_faced}
                  </p>
                  <p className="text-gray-700">
                    <span className="font-medium">Deals:</span> {metrics.successful_deals}
                  </p>
                </div>
              </div>
            ))}
          </div>

          {/* Emergent Behaviors */}
          {results.emergent_behaviors && results.emergent_behaviors.length > 0 && (
            <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
              <h4 className="font-bold text-blue-900 mb-2 flex items-center">
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
                Emergent Behaviors Detected
              </h4>
              <ul className="space-y-1">
                {results.emergent_behaviors.map((behavior, i) => (
                  <li key={i} className="text-sm text-blue-800">• {behavior}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Round-by-Round Analysis */}
          {results.rounds && currentRound > 0 && (
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="font-bold text-gray-800 mb-3">Round-by-Round Progress</h4>
              <div className="space-y-2">
                {results.rounds.slice(0, currentRound).map((round, i) => (
                  <div key={i} className="bg-white p-3 rounded border border-gray-200">
                    <p className="font-medium text-gray-700 mb-1">Round {i + 1}</p>
                    {round.competition_analysis?.strategies?.map((strategy, j) => (
                      <p key={j} className="text-sm text-gray-600">• {strategy}</p>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
