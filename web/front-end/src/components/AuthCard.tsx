import { useState } from "react";

export default function AuthCard() {
  const [flipped, setFlipped] = useState(false);

  return (
    <div className="w-[400px] h-[520px] [perspective:1000px]">
      <div
        className={`relative w-full h-full transition-transform duration-700 [transform-style:preserve-3d] ${
          flipped ? "[transform:rotateY(180deg)]" : ""
        }`}
      >
        {/* LOGIN CARD */}
        <div className="absolute w-full h-full bg-white rounded-3xl shadow-2xl p-8 [backface-visibility:hidden] flex flex-col">
          <h2 className="text-3xl font-bold text-center mb-8">Log In</h2>
          
          <div className="flex flex-col gap-4 flex-1">
            <input
              type="email"
              placeholder="Email"
              className="p-3 border border-gray-300 rounded-lg w-full focus:outline-none focus:border-purple-500 transition"
            />
            <input
              type="password"
              placeholder="Password"
              className="p-3 border border-gray-300 rounded-lg w-full focus:outline-none focus:border-purple-500 transition"
            />
            
            <button className="bg-black text-white w-full py-3 rounded-lg hover:bg-gray-800 transition font-semibold mt-2">
              Log In
            </button>

            <p className="text-center text-sm text-gray-600 mt-2">
              Already have an account?
            </p>

            <div className="text-center text-sm text-gray-500 my-2">Or</div>

            {/* Social Login Buttons */}
            <div className="flex gap-3 justify-center">
              <button className="w-12 h-12 border border-gray-300 rounded-lg flex items-center justify-center hover:bg-gray-50 transition">
                <svg className="w-5 h-5" viewBox="0 0 24 24">
                  <path fill="#1877F2" d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/>
                </svg>
              </button>
              
              <button className="w-12 h-12 border border-gray-300 rounded-lg flex items-center justify-center hover:bg-gray-50 transition">
                <svg className="w-5 h-5" viewBox="0 0 24 24">
                  <path fill="#EA4335" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                  <path fill="#4285F4" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                  <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                  <path fill="#34A853" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                </svg>
              </button>
              
              <button className="w-12 h-12 border border-gray-300 rounded-lg flex items-center justify-center hover:bg-gray-50 transition">
                <svg className="w-5 h-5" viewBox="0 0 24 24">
                  <path fill="#000" d="M12 2C6.477 2 2 6.477 2 12c0 4.991 3.657 9.128 8.438 9.879V14.89h-2.54V12h2.54V9.797c0-2.506 1.492-3.89 3.777-3.89 1.094 0 2.238.195 2.238.195v2.46h-1.26c-1.243 0-1.63.771-1.63 1.562V12h2.773l-.443 2.89h-2.33v6.989C18.343 21.129 22 16.99 22 12c0-5.523-4.477-10-10-10z"/>
                </svg>
              </button>
            </div>
          </div>

          <p className="text-center text-sm text-gray-600 mt-6">
            Don't have an account?{" "}
            <button
              onClick={() => setFlipped(true)}
              className="text-purple-600 font-semibold hover:underline"
            >
              Sign Up
            </button>
          </p>
        </div>

        {/* SIGN UP CARD */}
        <div className="absolute w-full h-full bg-white rounded-3xl shadow-2xl p-8 [backface-visibility:hidden] [transform:rotateY(180deg)] flex flex-col">
          <h2 className="text-3xl font-bold text-center mb-8">Sign Up</h2>
          
          <div className="flex flex-col gap-4 flex-1">
            <input
              type="text"
              placeholder="Name"
              className="p-3 border border-gray-300 rounded-lg w-full focus:outline-none focus:border-purple-500 transition"
            />
            <input
              type="email"
              placeholder="Email"
              className="p-3 border border-gray-300 rounded-lg w-full focus:outline-none focus:border-purple-500 transition"
            />
            <input
              type="password"
              placeholder="Password"
              className="p-3 border border-gray-300 rounded-lg w-full focus:outline-none focus:border-purple-500 transition"
            />
            
            <button className="bg-black text-white w-full py-3 rounded-lg hover:bg-gray-800 transition font-semibold mt-2">
              Sign Up
            </button>

            <div className="text-center text-sm text-gray-500 my-2">Or</div>

            {/* Social Signup Buttons */}
            <div className="flex gap-3 justify-center">
              <button className="w-12 h-12 border border-gray-300 rounded-lg flex items-center justify-center hover:bg-gray-50 transition">
                <svg className="w-5 h-5" viewBox="0 0 24 24">
                  <path fill="#1877F2" d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/>
                </svg>
              </button>
              
              <button className="w-12 h-12 border border-gray-300 rounded-lg flex items-center justify-center hover:bg-gray-50 transition">
                <svg className="w-5 h-5" viewBox="0 0 24 24">
                  <path fill="#EA4335" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                  <path fill="#4285F4" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                  <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                  <path fill="#34A853" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                </svg>
              </button>
              
              <button className="w-12 h-12 border border-gray-300 rounded-lg flex items-center justify-center hover:bg-gray-50 transition">
                <svg className="w-5 h-5" viewBox="0 0 24 24">
                  <path fill="#000" d="M12 2C6.477 2 2 6.477 2 12c0 4.991 3.657 9.128 8.438 9.879V14.89h-2.54V12h2.54V9.797c0-2.506 1.492-3.89 3.777-3.89 1.094 0 2.238.195 2.238.195v2.46h-1.26c-1.243 0-1.63.771-1.63 1.562V12h2.773l-.443 2.89h-2.33v6.989C18.343 21.129 22 16.99 22 12c0-5.523-4.477-10-10-10z"/>
                </svg>
              </button>
            </div>
          </div>

          <p className="text-center text-sm text-gray-600 mt-6">
            Already have an account?{" "}
            <button
              onClick={() => setFlipped(false)}
              className="text-purple-600 font-semibold hover:underline"
            >
              Log In
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}