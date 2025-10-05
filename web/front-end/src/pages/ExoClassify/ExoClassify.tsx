import { useState, useRef, useEffect } from "react";
import "./ExoClassify.css";
import { Link } from "react-router-dom";

// Import das imagens
import img1QST from "./images/img1QST.png";
import img2QST from "./images/img2QST.png";
// adicione mais imagens conforme necessário

// Adicione mais imagens aqui conforme necessário

type PracticeQuestion = {
  instruction: string;
  image: string;
};

// Array de perguntas
const questions: PracticeQuestion[] = [
  {
    instruction: "Draw a rectangle around the transit",
    image: img1QST,
  },
  {
    instruction: "Draw a rectangle around the transit",
    image: img2QST,
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
      const handleNext = () => {
    if (!drawnRect) {
      alert("Please draw a rectangle first!");
      return;
    }
    if (window.confirm("Are you sure you want to submit this selection and continue?")) {
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
                <>Thank you for your contribution!</>}
            </h1>
            <div className="result-message">
              {passed ? <p>Excellent work! You've mastered the drawing practice! Now you can start the real exploration 🚀</p> :
                <p>You’ve reached the end of the available images. Please check back later for more.</p>}
            </div>
            <div className="practice-nav">
              <Link to="/"><button className="ExoLearn">Go back</button></Link>
            
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
          <h1>Light Curve {currentStep + 1} </h1>
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
            <button onClick={handleNext}>
              {showFeedback ? (currentStep === questions.length - 1 ? "Finish" : "Next") : "Send Answer"}
            </button>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Practice;
