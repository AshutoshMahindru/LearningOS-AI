import React, { useState, useRef, useEffect } from 'react';
import { apiClient } from '../api/client';

interface SocraticDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  sessionId: string;
  stageId: string;
}

interface Message {
  role: 'user' | 'tutor';
  content: string;
}

export const SocraticDrawer: React.FC<SocraticDrawerProps> = ({ isOpen, onClose, sessionId, stageId }) => {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'tutor', content: 'I am your Socratic Tutor. I cannot give you the answer, but I can help you find it. What are you stuck on?' }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const handleSend = async () => {
    if (!input.trim()) return;
    
    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    try {
      const data = await apiClient.tutorChat(sessionId, stageId, { role: 'user', prompt: userMessage.content });
      setMessages(prev => [...prev, { role: 'tutor', content: data.response }]);
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, { role: 'tutor', content: 'Connection to tutor lost.' }]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed top-0 right-0 h-full w-96 glass-panel border-l border-slate-700/50 flex flex-col z-50 transform transition-transform duration-300 shadow-[-10px_0_30px_rgba(0,0,0,0.5)]">
      <div className="flex justify-between items-center p-5 border-b border-slate-700/50 bg-slate-900/40 backdrop-blur-md">
        <h2 className="text-lg font-bold text-primary flex items-center">
          <span className="mr-2 text-xl">🦉</span> Socratic Tutor
        </h2>
        <button onClick={onClose} className="text-slate-400 hover:text-white transition-colors">
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] p-3.5 text-sm shadow-md ${msg.role === 'user' ? 'bg-indigo-600 text-white rounded-2xl rounded-br-sm' : 'glass-card text-slate-200 rounded-2xl rounded-bl-sm'}`}>
              {msg.content}
            </div>
          </div>
        ))}
        {isTyping && (
          <div className="flex justify-start">
            <div className="glass-card text-slate-400 rounded-2xl rounded-bl-sm p-3.5 text-sm flex space-x-1 shadow-md">
              <span className="animate-bounce">.</span><span className="animate-bounce" style={{animationDelay: '0.1s'}}>.</span><span className="animate-bounce" style={{animationDelay: '0.2s'}}>.</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="p-4 border-t border-slate-700/50 bg-slate-900/60 backdrop-blur-md">
        <div className="flex rounded-xl bg-slate-950/50 border border-slate-700/50 overflow-hidden shadow-inner focus-within:border-indigo-500/50 transition-colors">
          <textarea
            className="flex-1 bg-transparent text-white p-3 text-sm focus:outline-none resize-none"
            rows={2}
            placeholder="Explain your problem..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button 
            onClick={handleSend}
            disabled={!input.trim() || isTyping}
            className="px-4 text-indigo-400 hover:bg-indigo-900/30 hover:text-indigo-300 disabled:opacity-50 transition-all flex items-center justify-center"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" /></svg>
          </button>
        </div>
      </div>
    </div>
  );
};
