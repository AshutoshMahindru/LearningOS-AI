import React, { useState, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import { apiClient } from '../api/client';
import { useAuth } from '../context/AuthContext';

export const OrientationStage: React.FC = () => <div className="p-4 border border-slate-700 rounded mb-4">Orientation Stage</div>;
export const TraceMapStage: React.FC = () => <div className="p-4 border border-slate-700 rounded mb-4">Trace Map Stage</div>;

export const InterrogateStage: React.FC<{sessionId: string, stageId: string}> = ({sessionId, stageId}) => {
  const [answer, setAnswer] = useState('');
  const [isSubmitted, setIsSubmitted] = useState(false);

  const handleSubmit = async () => {
    if (!answer.trim()) return;
    try {
      await apiClient.submitStage(sessionId, stageId, {
        artifacts: [{ type: "q_and_a", question: "Why is an array used here instead of a linked list?", answer: answer }],
        explanation: "Interrogation Answer"
      });
      setIsSubmitted(true);
      alert('Answer Submitted to Ledger!');
    } catch (err) {
      console.error(err);
      alert('Failed to submit answer.');
    }
  };

  return (
    <div className="bg-slate-800 p-6 rounded border border-slate-700 space-y-4">
      <h4 className="text-lg font-semibold text-primary mb-2">Interrogation: Mental Model Check</h4>
      <p className="text-slate-300">Why does the `process_data` function use a list append operation instead of pre-allocating an array?</p>
      
      <textarea
        className="w-full bg-slate-900 border border-slate-700 rounded p-3 text-sm text-white focus:outline-none focus:border-primary resize-none"
        rows={4}
        placeholder="Provide your reasoning..."
        value={answer}
        onChange={(e) => setAnswer(e.target.value)}
        disabled={isSubmitted}
      />
      
      <div className="flex justify-end">
        <button 
          onClick={handleSubmit}
          disabled={isSubmitted || !answer.trim()}
          className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white font-medium rounded disabled:opacity-50"
        >
          {isSubmitted ? 'Answer Submitted' : 'Submit Answer to Ledger'}
        </button>
      </div>
    </div>
  );
};
export const ExperimentStage: React.FC<{sessionId: string, stageId: string}> = ({sessionId, stageId}) => {
  const [hypothesis, setHypothesis] = useState('');
  const [isCommitted, setIsCommitted] = useState(false);
  const [code, setCode] = useState('# Write your experiment code here\n');
  const [result, setResult] = useState<any>(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);

  const handleCommit = async () => {
    if (!hypothesis.trim()) return;
    try {
      await apiClient.predictStage(sessionId, stageId, {
        hypothesis,
        expected_values: {}
      });
      setIsCommitted(true);
    } catch (err) {
      console.error(err);
      alert('Failed to commit prediction. Check backend connection.');
    }
  };

  const handleRun = async () => {
    setIsExecuting(true);
    setResult(null);
    try {
      const res = await apiClient.executeStage(sessionId, stageId, {
        code,
        parameters: {}
      });
      setResult(res);
    } catch (err) {
      console.error(err);
      setResult({ error: 'Execution failed.' });
    } finally {
      setIsExecuting(false);
    }
  };

  const handleSubmit = async () => {
    try {
      await apiClient.submitStage(sessionId, stageId, {
        artifacts: [{ type: "metric", value: result }],
        explanation: "Automatically submitted experiment."
      });
      setIsSubmitted(true);
      alert('Stage Submitted Successfully! Check Evidence Ledger.');
    } catch (err) {
      console.error(err);
      alert('Failed to submit stage.');
    }
  };

  return (
    <div className="bg-slate-800 p-4 rounded border border-slate-700 space-y-4">
      <div className="bg-slate-900 p-4 rounded border border-slate-700">
        <h4 className="text-lg font-semibold mb-2">1. Predict & Commit</h4>
        <textarea 
          className="w-full bg-slate-800 border border-slate-700 p-2 rounded text-slate-200 mb-2"
          rows={3}
          placeholder="State your hypothesis before running the code..."
          value={hypothesis}
          onChange={(e) => setHypothesis(e.target.value)}
          disabled={isCommitted}
        />
        <button 
          onClick={handleCommit}
          disabled={isCommitted || !hypothesis.trim()}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded disabled:opacity-50"
        >
          {isCommitted ? 'Prediction Sealed' : 'Commit Prediction'}
        </button>
      </div>

      <div className={`transition-opacity ${isCommitted ? 'opacity-100' : 'opacity-50 pointer-events-none'}`}>
        <h4 className="text-lg font-semibold mb-2">2. Execute Experiment</h4>
        <div className="h-64 rounded overflow-hidden border border-slate-700 mb-2">
          <Editor
            height="100%"
            defaultLanguage="python"
            theme="vs-dark"
            value={code}
            onChange={(val) => setCode(val || '')}
            options={{
              minimap: { enabled: false },
              readOnly: !isCommitted || isExecuting
            }}
          />
        </div>
        <div className="flex justify-end">
          <button 
            onClick={handleRun}
            disabled={!isCommitted || isExecuting}
            className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white font-medium rounded disabled:opacity-50"
          >
            {isExecuting ? 'Running...' : 'Run Code'}
          </button>
        </div>
      </div>

      {result && (
        <div className="bg-slate-900 p-4 rounded border border-slate-700">
          <h4 className="text-lg font-semibold mb-2">3. Observe Results</h4>
          <pre className="text-sm text-green-400 font-mono overflow-x-auto mb-4">
            {JSON.stringify(result, null, 2)}
          </pre>
          <div className="flex justify-end">
            <button 
              onClick={handleSubmit}
              disabled={isSubmitted}
              className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white font-medium rounded disabled:opacity-50"
            >
              {isSubmitted ? 'Stage Submitted' : 'Submit Stage to Ledger'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
export const CodeReadingStage: React.FC<{sessionId: string, stageId: string}> = ({sessionId, stageId}) => {
  const [notes, setNotes] = useState('');
  const [isSubmitted, setIsSubmitted] = useState(false);
  const sampleCode = `def process_data(data):
    # What does this function do?
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result`;

  const handleSubmit = async () => {
    if (!notes.trim()) return;
    try {
      await apiClient.submitStage(sessionId, stageId, {
        artifacts: [{ type: "reading_notes", value: notes }],
        explanation: "Code Reading Notes"
      });
      setIsSubmitted(true);
      alert('Notes Submitted to Ledger!');
    } catch (err) {
      console.error(err);
      alert('Failed to submit notes.');
    }
  };

  return (
    <div className="bg-slate-800 p-4 rounded border border-slate-700 flex flex-col md:flex-row gap-4">
      <div className="flex-1">
        <h4 className="text-lg font-semibold mb-2">Source Code</h4>
        <div className="border border-slate-700 rounded overflow-hidden h-64">
          <Editor
            height="100%"
            defaultLanguage="python"
            theme="vs-dark"
            value={sampleCode}
            options={{ readOnly: true, minimap: { enabled: false } }}
          />
        </div>
      </div>
      <div className="flex-1 flex flex-col">
        <h4 className="text-lg font-semibold mb-2">Your Explanation</h4>
        <textarea
          className="flex-1 bg-slate-900 border border-slate-700 rounded p-3 text-sm text-white focus:outline-none focus:border-primary resize-none mb-4"
          placeholder="Explain what the code does, identifying key invariants..."
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          disabled={isSubmitted}
        />
        <div className="flex justify-end">
          <button 
            onClick={handleSubmit}
            disabled={isSubmitted || !notes.trim()}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white font-medium rounded disabled:opacity-50"
          >
            {isSubmitted ? 'Notes Submitted' : 'Submit Notes to Ledger'}
          </button>
        </div>
      </div>
    </div>
  );
};
export const RebuildDebugStage: React.FC = () => <div className="p-4 border border-slate-700 rounded mb-4">Rebuild & Debug Stage</div>;
export const ControlledFailureStage: React.FC = () => <div className="p-4 border border-slate-700 rounded mb-4">Controlled Failure Stage</div>;
export const TransferAssessmentStage: React.FC = () => <div className="p-4 border border-slate-700 rounded mb-4">Transfer Assessment Stage</div>;

export const CompetencyGateStage: React.FC<{sessionId: string, stageId: string}> = () => {
  const [competencies, setCompetencies] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const { learnerId } = useAuth();

  useEffect(() => {
    if (!learnerId) return;
    apiClient.getCompetencies(learnerId)
      .then(data => {
        setCompetencies(data.competencies || []);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, [learnerId]);

  if (loading) return <div className="p-4 text-slate-500">Checking competency signatures...</div>;

  const hasRequiredCompetency = competencies.some(c => c.competency_id === 'comp.sys.hypothesis_testing' && c.level >= 1);

  return (
    <div className={`p-6 rounded border ${hasRequiredCompetency ? 'bg-green-900/20 border-green-700' : 'bg-red-900/20 border-red-700'}`}>
      <div className="flex items-center mb-4">
        <span className="text-2xl mr-3">{hasRequiredCompetency ? '🔓' : '🔒'}</span>
        <h4 className={`text-xl font-bold ${hasRequiredCompetency ? 'text-green-400' : 'text-red-400'}`}>
          Competency Gate
        </h4>
      </div>
      
      {hasRequiredCompetency ? (
        <div>
          <p className="text-green-300 mb-2">Cryptographic evidence verified. You possess the required mental models to proceed.</p>
          <button className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white font-medium rounded transition-colors">
            Unlock Next Phase
          </button>
        </div>
      ) : (
        <div>
          <p className="text-red-300 mb-2">Missing required evidence for <code className="bg-red-900/50 px-1 rounded">comp.sys.hypothesis_testing</code> (Level 1).</p>
          <p className="text-sm text-slate-400">Complete the preceding Experiment Stage and submit your findings to the ledger to unlock this gate.</p>
        </div>
      )}
    </div>
  );
};
export const ReflectionADRStage: React.FC = () => <div className="p-4 border border-slate-700 rounded mb-4">Reflection & ADR Stage</div>;
export const FlagshipIntegrationStage: React.FC = () => <div className="p-4 border border-slate-700 rounded mb-4">Flagship Integration Stage</div>;
