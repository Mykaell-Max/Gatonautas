import { useEffect, useState } from "react";
import { Link } from 'react-router-dom';
import "./Quiz.css";

type Option = {
  text: string;
  isCorrect: boolean;
};

type Question = {
  question: string;
  options: Option[];
};

const Quiz = () => {
  const [questions, setQuestions] = useState<Question[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [showFeedback, setShowFeedback] = useState(false);
  const [answers, setAnswers] = useState<boolean[]>([]);
  const [quizFinished, setQuizFinished] = useState(false);

  useEffect(() => {
    import('./quiz.txt?raw')
      .then((module) => {
        const text = module.default;
        
        const lines = text
          .split("\n")
          .map(line => line.trim())
          .filter(line => line !== "");
        
        const parsed: Question[] = [];

        for (let i = 0; i < lines.length; i += 5) {
          if (i + 4 < lines.length) {
            const options: Option[] = [];
            
            for (let j = 1; j <= 4; j++) {
              const optionLine = lines[i + j];
              const isCorrect = optionLine.includes("|correct");
              const text = optionLine.replace("|correct", "").trim();
              
              options.push({ text, isCorrect });
            }
            
            parsed.push({
              question: lines[i],
              options
            });
          }
        }

        setQuestions(parsed);
      })
      .catch((error) => {
        console.error("Erro ao carregar quiz.txt:", error);
      });
  }, []);

  const handleOptionSelect = (index: number) => {
    if (!showFeedback) {
      setSelectedAnswer(index);
    }
  };

  const handleNext = () => {
    if (selectedAnswer === null) {
      alert("Por favor, selecione uma resposta antes de continuar!");
      return;
    }

    if (!showFeedback) {
      setShowFeedback(true);
      
      const isCorrect = questions[currentStep].options[selectedAnswer].isCorrect;
      setAnswers([...answers, isCorrect]);
    } else {
      if (currentStep === questions.length - 1) {
        setQuizFinished(true);
      } else {
        setCurrentStep(currentStep + 1);
        setSelectedAnswer(null);
        setShowFeedback(false);
      }
    }
  };

  const handlePrev = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
      setSelectedAnswer(null);
      setShowFeedback(false);
      setAnswers(answers.slice(0, -1));
    }
  };

  const handleRestart = () => {
    setCurrentStep(0);
    setSelectedAnswer(null);
    setShowFeedback(false);
    setAnswers([]);
    setQuizFinished(false);
  };

  const getOptionClass = (index: number, option: Option) => {
    if (!showFeedback) {
      return "option-box";
    }

    if (option.isCorrect) {
      return "option-box option-correct";
    }

    if (selectedAnswer === index && !option.isCorrect) {
      return "option-box option-incorrect";
    }

    return "option-box";
  };

  const calculateScore = () => {
    const correctAnswers = answers.filter(answer => answer).length;
    const percentage = Math.round((correctAnswers / questions.length) * 100);
    return { correctAnswers, percentage };
  };

  if (questions.length === 0) {
    return (
      <div className="quiz-container">
        <section className="quiz-content">
          <div className="quiz-text">
            <p className="question">Loading questions...</p>
          </div>
        </section>
      </div>
    );
  }

  if (quizFinished) {
    const { correctAnswers, percentage } = calculateScore();
    const passed = percentage >= 50;

    return (
      <div className="quiz-container">
        <section className="quiz-content">
          <div className="quiz-text">
            <h1 className="result-title">
              {passed ? (
                <>Congrats! <span className="highlight">You Passed!</span></>
              ) : (
                <>Keep <span className="highlight">Learning!</span></>
              )}
            </h1>
            
            <div className="result-score">
              <div className="score-circle">
                <span className="score-percentage">{percentage}%</span>
                <span className="score-label">Score</span>
              </div>
            </div>

            <p className="result-details">
              You answered {correctAnswers} out of {questions.length} questions correctly
            </p>

            <div className="result-message">
              {passed ? (
                <p>
                  You passed the training step, now you can learn by doing it!
                </p>
              ) : (
                <p>
                  You need at least 50% to pass. Go back to the training and study more before trying again!
                </p>
              )}
            </div>

            <div className="quiz-nav">
              <button onClick={handleRestart}>
                Try Again
              </button>
              {passed && (
                <Link to="/Explore">
                    <button className="Explore">Start Explore</button>
                </Link>
              )}
            </div>
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="quiz-container">
      <section className="quiz-content">
        <div className="quiz-text">
          <h1>
            Step {currentStep + 1}: <span className="highlight">Theory</span>
          </h1>
          
          <p className="question">{questions[currentStep].question}</p>
          <div className="options">
            {questions[currentStep].options.map((opt, idx) => (
              <label 
                key={idx} 
                className={getOptionClass(idx, opt)}
                onClick={() => handleOptionSelect(idx)}
              >
                <input 
                  type="radio" 
                  name={`q${currentStep}`}
                  checked={selectedAnswer === idx}
                  onChange={() => handleOptionSelect(idx)}
                  disabled={showFeedback}
                />
                <span>{opt.text}</span>
              </label>
            ))}
          </div>
          
          <div className="quiz-nav">
            <button 
              onClick={handlePrev} 
              disabled={currentStep === 0}
            >
              Previous
            </button>
            <button 
              onClick={handleNext}
            >
              {showFeedback ? (currentStep === questions.length - 1 ? "Finish" : "Next") : "Check Answer"}
            </button>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Quiz;