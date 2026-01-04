import React from "react";

const plans = [
  {
    name: "Free",
    price: 0,
    unit: "/month",
    description: "For getting started",
    features: ["5 daily credits", "Public projects"],
    button: <button className="bg-gray-700 text-white rounded px-4 py-2 mt-4 cursor-not-allowed" disabled>当前套餐</button>,
  },
  {
    name: "Pro",
    price: 25,
    unit: "/month",
    description: "For more projects and usage",
    features: [
      "100 credits / month",
      "Private projects",
      "Remove the Lovable badge",
      "Custom domains",
      "3 editors per project",
    ],
    button: (
      <div className="flex flex-col gap-2 mt-4">
        <button className="bg-blue-600 hover:bg-blue-700 text-white rounded px-4 py-2">微信支付</button>
        <button className="bg-green-500 hover:bg-green-600 text-white rounded px-4 py-2">支付宝支付</button>
      </div>
    ),
    popular: true,
  },
  {
    name: "Teams",
    price: 30,
    unit: "/month",
    description: "For collaborating with others",
    features: [
      "Centralised billing",
      "Centralised access management",
      "Includes 20 seats",
      "Everything in Pro, plus:",
    ],
    button: (
      <div className="flex flex-col gap-2 mt-4">
        <button className="bg-blue-600 hover:bg-blue-700 text-white rounded px-4 py-2">微信支付</button>
        <button className="bg-green-500 hover:bg-green-600 text-white rounded px-4 py-2">支付宝支付</button>
      </div>
    ),
  },
];

const Pricing: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#18181b] flex flex-col items-center justify-center p-6">
      <div className="max-w-3xl w-full">
        <h1 className="text-3xl font-bold text-white mb-2">Building something big?</h1>
        <p className="text-gray-400 mb-8">升级套餐，获得更多项目额度和团队协作能力。</p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {plans.map((plan) => (
            <div
              key={plan.name}
              className={`rounded-xl bg-[#232329] p-6 flex flex-col items-center border border-gray-700 relative ${plan.popular ? "border-blue-600" : ""}`}
            >
              {plan.popular && (
                <span className="absolute top-4 right-4 bg-blue-600 text-white text-xs px-2 py-1 rounded">POPULAR</span>
              )}
              <h2 className="text-xl font-bold text-white mb-2">{plan.name}</h2>
              <div className="flex items-end mb-2">
                <span className="text-3xl font-bold text-white">${plan.price}</span>
                <span className="text-gray-400 ml-1">{plan.unit}</span>
              </div>
              <p className="text-gray-400 mb-4">{plan.description}</p>
              <ul className="text-gray-300 text-sm mb-4 space-y-1 list-disc list-inside">
                {plan.features.map((f, i) => (
                  <li key={i}>{f}</li>
                ))}
              </ul>
              {plan.button}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Pricing; 