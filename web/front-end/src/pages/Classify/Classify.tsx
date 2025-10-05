import { useState, useRef, useEffect } from "react";
import "./Classify.css";
import { Link } from "react-router-dom";

// Import das imagens
import img1QST from "./images/img1QST.png";
import img2QST from "./images/img2QST.png";
// Adicione mais imagens aqui conforme necessário

type PracticeQuestion = {
  instruction: string;
  image: string;
  correctArea: { x1: number; y1: number; x2: number; y2: number };
};

// Array de perguntas
const questions: PracticeQuestion[] = [
  {
    instruction: "Draw a rectangle around the transit",
    image: img1QST,
    correctArea: { x1: 400.5078125, y1: 40.0859375, x2: 542.5078125, y2: 309.0859375 },
  },
  {
    instruction: "Draw a rectangle around the transit",
    image: img2QST,
    correctArea: { x1: 221.1328125  , y1: 46.5703125  , x2: 277.1328125  , y2: 302.5703125 },
  },
  // continue adicionando todas aqui
];

const Practice = () => {
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

  // Desenho no canvas
  useEffect(() => {
    if (canvasRef.current && imageRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      const img = imageRef.current;

      if (ctx && img.complete) {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Retângulo sendo desenhado
        if (currentRect) {
          ctx.strokeStyle = "#ffcc00";
          ctx.lineWidth = 3;
          ctx.strokeRect(currentRect.x, currentRect.y, currentRect.width, currentRect.height);
        }

        // Retângulo final
        if (drawnRect) {
          ctx.strokeStyle = showFeedback ? (isCorrect ? "#22c55e" : "#ef4444") : "#ffcc00";
          ctx.lineWidth = 3;
          ctx.strokeRect(drawnRect.x, drawnRect.y, drawnRect.width, drawnRect.height);
        }

        // Retângulo correto se errou
        if (showFeedback && !isCorrect) {
          ctx.strokeStyle = "#22c55e";
          ctx.lineWidth = 3;
          ctx.setLineDash([5, 5]);
          const correct = questions[currentStep].correctArea;
          ctx.strokeRect(correct.x1, correct.y1, correct.x2 - correct.x1, correct.y2 - correct.y1);
          ctx.setLineDash([]);
        }
      }
    }
  }, [currentRect, drawnRect, showFeedback, isCorrect, currentStep]);

  // Eventos do mouse
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

  // Checagem de resposta
  const checkAnswer = () => {
    if (!drawnRect) {
      alert("Please draw a rectangle first!");
      return;
    }
    const correct = questions[currentStep].correctArea;
    const drawn = {
      x1: drawnRect.x,
      y1: drawnRect.y,
      x2: drawnRect.x + drawnRect.width,
      y2: drawnRect.y + drawnRect.height
    };
    console.log(drawn.x1,drawn.y1,drawn.x2,drawn.y2)
    const overlapX = Math.max(0, Math.min(drawn.x2, correct.x2) - Math.max(drawn.x1, correct.x1));
    const overlapY = Math.max(0, Math.min(drawn.y2, correct.y2) - Math.max(drawn.y1, correct.y1));
    const overlapArea = overlapX * overlapY;
    const drawnArea = drawnRect.width * drawnRect.height;
    const correctArea = (correct.x2 - correct.x1) * (correct.y2 - correct.y1);
    const iou = overlapArea / (drawnArea + correctArea - overlapArea);
    const correct_answer = iou > 0.6;
    setIsCorrect(correct_answer);
    setShowFeedback(true);
    setAnswers([...answers, correct_answer]);
  };

  // Navegação
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
    const correctAnswers = answers.filter(a => a).length;
    const percentage = Math.round((correctAnswers / questions.length) * 100);
    return { correctAnswers, percentage };
  };

  // Render
  if (practiceFinished) {
    const { correctAnswers, percentage } = calculateScore();
    const passed = percentage >= 50;
    return (
      <div className="practice-container">
        <section className="practice-content">
          <div className="practice-text">
            <h1 className="result-title">
              {passed ? <>Congrats! <span className="highlight">You Passed!</span></> :
                <>Keep <span className="highlight">Practicing!</span></>}
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
              {passed ? <p>Excellent work! You've mastered the drawing practice! Now you can start the real exploration 🚀</p> :
                <p>Keep practicing to improve your accuracy. Try again!</p>}
            </div>
            <div className="practice-nav">
              {isCorrect ? <Link to="/Explore"><button className="explore">Explore</button></Link> : <button onClick={handleRestart}>Try Again</button>}
            </div>
          </div>
        </section>
      </div>
    );
  }

  // Main practice render
  return (
    <div className="practice-container">
      <section className="practice-content">
        <div className="practice-text">
          <h1>Step {currentStep + 1}: <span className="highlight">Practice</span></h1>
          <p className="instruction">{questions[currentStep].instruction}</p>
          <div className="canvas-wrapper">
            <img
              ref={imageRef}
              src={questions[currentStep].image}
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
            <button onClick={handlePrev} disabled={currentStep === 0}>Previous</button>
            <button onClick={handleNext}>
              {showFeedback ? (currentStep === questions.length - 1 ? "Finish" : "Next") : "Check Answer"}
            </button>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Practice;
