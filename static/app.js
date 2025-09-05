document.getElementById("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = document.getElementById("user-input").value;
  if (!input.trim()) return;

  const chatBox = document.getElementById("chat-box");
  chatBox.innerHTML += `<div><strong>You:</strong> ${input}</div>`;

  const response = await fetch("/get", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: `msg=${encodeURIComponent(input)}`
  });

  const data = await response.json();
  chatBox.innerHTML += `<div><strong>Bot:</strong> ${data.answer}</div>`;
  document.getElementById("user-input").value = "";
  chatBox.scrollTop = chatBox.scrollHeight;
});