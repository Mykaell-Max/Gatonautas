import { useEffect, useState } from "react";
import "./Quiz.css";
import QuizImage from "../../assets/Quiz.svg";

type Question = {
  question: string;
  options: string[];
};

const Quiz= () => {
  const [questions, setQuestions] = useState<Question[]>([]);
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    fetch("/questions.txt")
      .then((res) => res.text())
      .then((text) => {
        const lines = text.split("\n").filter((line) => line.trim() !== "");
        const parsed: Question[] = [];

        for (let i = 0; i < lines.length; i += 5) {
          parsed.push({
            question: lines[i],
            options: lines.slice(i + 1, i + 5),
          });
        }

        setQuestions(parsed);
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
          {questions.length > 0 && (
            <>
              <p className="question">{questions[currentStep].question}</p>
              <div className="options">
                {questions[currentStep].options.map((opt, idx) => (
                  <label key={idx} className="option-box">
                    <input type="radio" name={`q${currentStep}`} />
                    <span>{opt}</span>
                  </label>
                ))}
              </div>
            </>
          )}
          <div className="quiz-nav">
            <button onClick={handlePrev} disabled={currentStep === 0}>
              Previous
            </button>
            <button onClick={handleNext} disabled={currentStep === questions.length - 1}>
              Next
            </button>
          </div>
        </div>
      </section>

      <img src={QuizImage} alt="Quiz planet" className="quiz-image" />
    </div>
  );
};
export default Quiz;
