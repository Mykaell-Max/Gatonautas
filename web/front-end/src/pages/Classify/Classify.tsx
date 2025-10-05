import { useEffect, useState, useRef } from "react";
import "./Classify.css";

type PracticeQuestion = {
  instruction: string;
  imagePath: string;
  correctArea: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
};

const Practice = () => {
  const [questions, setQuestions] = useState<PracticeQuestion[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPoint, setStartPoint] = useState<{ x: number; y: number } | null>(null);
  const [currentRect, setCurrentRect] = useState<{ x: number; y: number; width: number; height: number } | null>(null);
  const [drawnRect, setDrawnRect] = useState<{ x: number; y: number; width: number; height: number } | null>(null);
  const [showFeedback, setShowFeedback] = useState(false);
  const [isCorrect, setIsCorrect] = useState(false);
  const [answers, setAnswers] = useState<boolean[]>([]);
  const [practiceFinished, setPracticeFinished] = useState(false);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    import('./practice.txt?raw')
      .then((module) => {
        const text = module.default;
        
        const lines = text
          .split("\n")
          .map(line => line.trim())
          .filter(line => line !== "");
        
        const parsed: PracticeQuestion[] = [];

        for (let i = 0; i < lines.length; i += 3) {
          if (i + 2 < lines.length) {
            const coords = lines[i + 2].split(",").map(Number);
            
            parsed.push({
              instruction: lines[i],
              imagePath: lines[i + 1],
              correctArea: {
                x1: coords[0],
                y1: coords[1],
                x2: coords[2],
                y2: coords[3]
              }
            });
          }
        }

        setQuestions(parsed);
      })
      .catch((error) => {
        console.error("Erro ao carregar practice.txt:", error);
      });
  }, []);

  useEffect(() => {
    if (canvasRef.current && imageRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      const img = imageRef.current;

      if (ctx && img.complete) {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Desenha o retângulo sendo desenhado
        if (currentRect) {
          ctx.strokeStyle = "#ffcc00";
          ctx.lineWidth = 3;
          ctx.strokeRect(currentRect.x, currentRect.y, currentRect.width, currentRect.height);
        }

        // Desenha o retângulo final
        if (drawnRect) {
          ctx.strokeStyle = showFeedback ? (isCorrect ? "#22c55e" : "#ef4444") : "#ffcc00";
          ctx.lineWidth = 3;
          ctx.strokeRect(drawnRect.x, drawnRect.y, drawnRect.width, drawnRect.height);
        }

        // Mostra o retângulo correto se errou
        if (showFeedback && !isCorrect && questions[currentStep]) {
          ctx.strokeStyle = "#22c55e";
          ctx.lineWidth = 3;
          ctx.setLineDash([5, 5]);
          const correct = questions[currentStep].correctArea;
          ctx.strokeRect(
            correct.x1,
            correct.y1,
            correct.x2 - correct.x1,
            correct.y2 - correct.y1
          );
          ctx.setLineDash([]);
        }
      }
    }
  }, [currentRect, drawnRect, showFeedback, isCorrect, currentStep, questions]);

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (showFeedback || !canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setIsDrawing(true);
    setStartPoint({ x, y });
    setDrawnRect(null);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !startPoint || !canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setCurrentRect({
      x: Math.min(startPoint.x, x),
      y: Math.min(startPoint.y, y),
      width: Math.abs(x - startPoint.x),
      height: Math.abs(y - startPoint.y)
    });
  };

  const handleMouseUp = () => {
    if (!isDrawing || !currentRect) return;

    setIsDrawing(false);
    setDrawnRect(currentRect);
    setCurrentRect(null);
  };

  const checkAnswer = () => {
    if (!drawnRect) {
      alert("Por favor, desenhe um retângulo antes de verificar!");
      return;
    }

    const correct = questions[currentStep].correctArea;
    const drawn = {
      x1: drawnRect.x,
      y1: drawnRect.y,
      x2: drawnRect.x + drawnRect.width,
      y2: drawnRect.y + drawnRect.height
    };

    // Calcula a sobreposição (IoU simplificado)
    const overlapX = Math.max(0, Math.min(drawn.x2, correct.x2) - Math.max(drawn.x1, correct.x1));
    const overlapY = Math.max(0, Math.min(drawn.y2, correct.y2) - Math.max(drawn.y1, correct.y1));
    const overlapArea = overlapX * overlapY;

    const drawnArea = drawnRect.width * drawnRect.height;
    const correctArea = (correct.x2 - correct.x1) * (correct.y2 - correct.y1);
    const unionArea = drawnArea + correctArea - overlapArea;

    const iou = overlapArea / unionArea;
    const correct_answer = iou > 0.5; // 50% de sobreposição

    setIsCorrect(correct_answer);
    setShowFeedback(true);
    setAnswers([...answers, correct_answer]);
  };

  const handleNext = () => {
    if (!showFeedback) {
      checkAnswer();
    } else {
      if (currentStep === questions.length - 1) {
        setPracticeFinished(true);
      } else {
        setCurrentStep(currentStep + 1);
        setDrawnRect(null);
        setShowFeedback(false);
        setIsCorrect(false);
      }
    }
  };

  const handlePrev = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
      setDrawnRect(null);
      setShowFeedback(false);
      setIsCorrect(false);
      setAnswers(answers.slice(0, -1));
    }
  };

  const handleRestart = () => {
    setCurrentStep(0);
    setDrawnRect(null);
    setShowFeedback(false);
    setIsCorrect(false);
    setAnswers([]);
    setPracticeFinished(false);
  };

  const calculateScore = () => {
    const correctAnswers = answers.filter(answer => answer).length;
    const percentage = Math.round((correctAnswers / questions.length) * 100);
    return { correctAnswers, percentage };
  };

  if (questions.length === 0) {
    return (
      <div className="practice-container">
        <section className="practice-content">
          <div className="practice-text">
            <p className="instruction">Loading practice...</p>
          </div>
        </section>
      </div>
    );
  }

  if (practiceFinished) {
    const { correctAnswers, percentage } = calculateScore();
    const passed = percentage >= 50;

    return (
      <div className="practice-container">
        <section className="practice-content">
          <div className="practice-text">
            <h1 className="result-title">
              {passed ? (
                <>Congrats! <span className="highlight">You Passed!</span></>
              ) : (
                <>Keep <span className="highlight">Practicing!</span></>
              )}
            </h1>
            
            <div className="result-score">
              <div className="score-circle">
                <span className="score-percentage">{percentage}%</span>
                <span className="score-label">Score</span>
              </div>
            </div>

            <p className="result-details">
              You completed {correctAnswers} out of {questions.length} tasks correctly
            </p>

            <div className="result-message">
              {passed ? (
                <p>
                  Excellent work! You've mastered the drawing practice!
                </p>
              ) : (
                <p>
                  Keep practicing to improve your accuracy. Try again!
                </p>
              )}
            </div>

            <div className="practice-nav">
              <button onClick={handleRestart}>
                Try Again
              </button>
            </div>
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="practice-container">
      <section className="practice-content">
        <div className="practice-text">
          <h1>
            Step {currentStep + 1}: <span className="highlight">Practice</span>
          </h1>
          
          <p className="instruction">{questions[currentStep].instruction}</p>
          
          <div className="canvas-wrapper">
            <img
              ref={imageRef}
              src={`/${questions[currentStep].imagePath}`}
              alt="Practice"
              className="practice-image"
              onLoad={() => {
                if (canvasRef.current && imageRef.current) {
                  canvasRef.current.width = imageRef.current.width;
                  canvasRef.current.height = imageRef.current.height;
                }
              }}
            />
            <canvas
              ref={canvasRef}
              className="drawing-canvas"
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
            />
          </div>

          {showFeedback && (
            <div className={`feedback-box ${isCorrect ? "correct" : "incorrect"}`}>
              {isCorrect ? "Correct! Well done!" : "Not quite right. The correct area is shown in green dashed line."}
            </div>
          )}
          
          <div className="practice-nav">
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

export default Practice;