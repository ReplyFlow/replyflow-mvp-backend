// ReplyFlow Chrome extension popup script
// When the user clicks the "Open Dashboard" button, open the ReplyFlow login
// page in a new tab.  You can update the URL below to point to your final
// production domain once the SSL certificate is configured and the site is
// publicly available.

// Set the URL of the ReplyFlow login page.  Update this value to point
// to your production domain once SSL is configured.  If you change the
// domain in your deployment (for example, to replyflowapp.com), you should
// also update this constant accordingly.
const dashboardUrl = "https://replyflowapp.com/login.html";

document.getElementById('openDashboard').addEventListener('click', () => {
  chrome.tabs.create({ url: dashboardUrl });
});