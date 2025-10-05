import { useEffect, useState } from "react";
import "./Quiz.css";

type Question = {
  question: string;
  options: string[];
  isCode?: boolean;
};

const Quiz = () => {
  const [questions, setQuestions] = useState<Question[]>([]);
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    fetch("/quiz.txt")
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.text();
      })
      .then((text) => {
        console.log("Conteúdo bruto do arquivo:", text);
        
        // Verifica se retornou HTML em vez de texto
        if (text.includes("<!DOCTYPE") || text.includes("<!doctype")) {
          throw new Error("O arquivo quiz.txt não foi encontrado. Retornou HTML em vez de texto.");
        }
        
        const lines = text
          .split("\n")
          .map(line => line.trim())
          .filter(line => line !== "" && !line.startsWith("<") && !line.includes("html"));
        
        console.log("Linhas filtradas:", lines);
        
        if (lines.length === 0) {
          throw new Error("O arquivo quiz.txt está vazio ou tem formato incorreto.");
        }
        
        const parsed: Question[] = [];

        // Processa perguntas em grupos de 5 linhas (1 pergunta + 4 opções)
        for (let i = 0; i < lines.length; i += 5) {
          if (i + 4 < lines.length) {
            parsed.push({
              question: lines[i],
              options: [
                lines[i + 1],
                lines[i + 2],
                lines[i + 3],
                lines[i + 4]
              ]
            });
          }
        }

        if (parsed.length === 0) {
          throw new Error("Nenhuma pergunta foi processada. Verifique o formato do arquivo.");
        }

        console.log("Perguntas carregadas:", parsed);
        setQuestions(parsed);
      })
      .catch((error) => {
        console.error("Erro ao carregar quiz.txt:", error);
        alert(`ERRO: ${error.message}\n\nVerifique:\n1. O arquivo quiz.txt existe na pasta public/\n2. O arquivo tem o formato correto\n3. Reinicie o servidor de desenvolvimento`);
      });
  }, []);

  const handleNext = () => {
    if (currentStep < questions.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrev = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  return (
    <div className="quiz-container">
      <section className="quiz-content">
        <div className="quiz-text">
          <h1>
            Step {currentStep + 1}: <span className="highlight">Theory</span>
          </h1>
          
          {questions.length > 0 ? (
            <>
              {questions[currentStep].isCode ? (
                <pre className="code-block">
                  {questions[currentStep].question}
                </pre>
              ) : (
                <>
                  <p className="question">{questions[currentStep].question}</p>
                  <div className="options">
                    {questions[currentStep].options?.map((opt, idx) => (
                      <label key={idx} className="option-box">
                        <input 
                          type="radio" 
                          name={`q${currentStep}`}
                        />
                        <span>{opt}</span>
                      </label>
                    ))}
                  </div>
                </>
              )}
            </>
          ) : (
            <p className="question">Loading questions...</p>
          )}
          
          <div className="quiz-nav">
            <button 
              onClick={handlePrev} 
              disabled={currentStep === 0}
            >
              Previous
            </button>
            <button 
              onClick={handleNext} 
              disabled={currentStep === questions.length - 1}
            >
              Next
            </button>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Quiz;