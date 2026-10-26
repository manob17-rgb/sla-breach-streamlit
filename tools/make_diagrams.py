import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle, Polygon

def actor(ax, x, y, label, color="#1f3a5f"):
    ax.add_patch(plt.Circle((x, y+0.55), 0.17, fill=False, ec=color, lw=2))
    ax.plot([x, x], [y+0.38, y-0.15], color=color, lw=2)
    ax.plot([x-0.3, x+0.3], [y+0.22, y+0.22], color=color, lw=2)
    ax.plot([x, x-0.25], [y-0.15, y-0.6], color=color, lw=2)
    ax.plot([x, x+0.25], [y-0.15, y-0.6], color=color, lw=2)
    ax.text(x, y-0.9, label, ha="center", va="top", fontsize=10.5, fontweight="bold", color=color)

def usecase(ax, x, y, text, w=2.9, h=0.78, fc="#e8f1fb", ec="#1f3a5f"):
    ax.add_patch(Ellipse((x, y), w, h, fc=fc, ec=ec, lw=1.4, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=8.6, zorder=4)
    return (x, y)

def line(ax, a, b, style="-", color="#444", label=None, arrow=False, rad=0.0, lo=(0, 0.12)):
    ax.annotate("", xy=b, xytext=a, zorder=2,
                arrowprops=dict(arrowstyle="-|>" if arrow else "-", ls=style, color=color, lw=1.2,
                                connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        ax.text((a[0]+b[0])/2+lo[0], (a[1]+b[1])/2+lo[1], label, fontsize=7.6, style="italic",
                color=color, ha="center", zorder=5, bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))

# ------------------------------------------------------------------ use case
fig, ax = plt.subplots(figsize=(15, 9.8)); ax.set_xlim(0, 15); ax.set_ylim(-0.1, 9.7); ax.axis("off")
ax.text(7.5, 9.45, "Updated Use Case Diagram - Student Service Request Monitoring (Streamlit deployment)",
        ha="center", fontsize=13.5, fontweight="bold")
ax.add_patch(Rectangle((3.0, 0.35), 9.0, 8.75, fc="#fbfbfd", ec="#1f3a5f", lw=1.8, zorder=0))
ax.text(3.15, 9.0, "Request Monitoring System (Streamlit app)", fontsize=9.5, fontweight="bold", va="top", color="#1f3a5f")
student = (1.2, 7.4); staff = (1.2, 3.6); admin = (13.7, 5.3)
actor(ax, *student, "Student"); actor(ax, *staff, "Student Services\nStaff"); actor(ax, *admin, "System\nAdministrator")
U = {}
U["status"]  = usecase(ax, 5.0, 8.0, "Check Request Status")
U["submit"]  = usecase(ax, 5.0, 6.9, "Submit Service Request")
U["predict"] = usecase(ax, 5.0, 5.8, "Predict SLA Breach Risk\n(submission-time inputs only)", w=3.5, h=0.95, fc="#fff1d6")
U["flag"]    = usecase(ax, 5.0, 4.85, "Flag High-Risk Request", fc="#fff1d6")
U["review"]  = usecase(ax, 5.0, 3.7, "Review Flagged Request", fc="#fff1d6")
U["decide"]  = usecase(ax, 5.0, 2.55, "Record Final Human Decision", w=3.3, fc="#fff1d6")
U["upload"]  = usecase(ax, 10.2, 8.0, "Upload & Validate Dataset", fc="#fff1d6")
U["eda"]     = usecase(ax, 10.2, 6.9, "View Data Analysis (EDA)", fc="#fff1d6")
U["dash"]    = usecase(ax, 10.2, 5.2, "View Decision Dashboard\n(KPIs, review queue)", w=3.3, h=0.95, fc="#fff1d6")
U["export"]  = usecase(ax, 10.2, 3.7, "Export Decision Log", fc="#fff1d6")
U["retrain"] = usecase(ax, 10.2, 2.55, "Update / Retrain Model", fc="#fff1d6")
# associations
for u in ("submit", "status"): line(ax, (student[0]+0.3, student[1]+0.2), (U[u][0]-1.45, U[u][1]))
for u, x in (("predict", 1.75), ("review", 1.45), ("decide", 1.65)):
    line(ax, (staff[0]+0.3, staff[1]+0.2), (U[u][0]-x, U[u][1]), color="#2e7d32")
line(ax, (staff[0]+0.3, staff[1]+0.2), (U["dash"][0]-1.65, U["dash"][1]-0.1), color="#2e7d32")
for u in ("upload", "eda", "dash", "export", "retrain"):
    w = 1.65 if u == "dash" else 1.45
    line(ax, (admin[0]-0.3, admin[1]+0.2), (U[u][0]+w, U[u][1]), color="#8a3b12")
# dependencies
line(ax, (5.0, U["submit"][1]-0.39), (5.0, U["predict"][1]+0.475), "--", arrow=True, label="<<triggers>>", lo=(0.75, 0))
line(ax, (5.0, U["predict"][1]-0.475), (5.0, U["flag"][1]+0.39), "--", arrow=True, label="<<extend>> risk >= 50%", lo=(1.15, 0))
line(ax, (5.0, U["flag"][1]-0.39), (5.0, U["review"][1]+0.39), "--", arrow=True, label="<<include>>", lo=(0.75, 0))
line(ax, (5.0, U["review"][1]-0.39), (5.0, U["decide"][1]+0.39), "--", arrow=True, label="<<include>>", lo=(0.75, 0))
line(ax, (10.2, U["upload"][1]-0.39), (10.2, U["eda"][1]+0.39), "--", arrow=True, label="<<include>>", lo=(0.75, 0))
line(ax, (U["predict"][0]+1.75, U["predict"][1]), (U["dash"][0]-1.65, U["dash"][1]+0.1), "--", arrow=True, label="risk scores feed", lo=(0, 0.18))
line(ax, (U["decide"][0]+1.65, U["decide"][1]+0.1), (U["export"][0]-1.2, U["export"][1]-0.25), "--", arrow=True, label="<<extend>>", lo=(-0.1, -0.3))
line(ax, (U["decide"][0]+1.65, U["decide"][1]-0.05), (U["retrain"][0]-1.45, U["retrain"][1]-0.05), "--", arrow=True, label="outcomes feed retraining", lo=(0, -0.28))
# generalization: Administrator is-a Staff (routed under the boundary)
gx = [admin[0], admin[0], staff[0], staff[0]]; gy = [admin[1]-1.5, 0.12, 0.12, staff[1]-1.3]
ax.plot(gx[:3], gy[:3], color="#8a3b12", lw=1.3, ls="-"); ax.plot(gx[2:], gy[2:], color="#8a3b12", lw=1.3)
ax.annotate("", xy=(staff[0], staff[1]-1.25), xytext=(staff[0], staff[1]-1.5),
            arrowprops=dict(arrowstyle="-|>", fc="white", ec="#8a3b12", lw=1.3, mutation_scale=16))
ax.text(7.5, 0.0, "Administrator inherits every Staff use case (generalization)", fontsize=8, style="italic", ha="center", color="#8a3b12")
# notes
ax.add_patch(FancyBboxPatch((3.25, 0.55), 4.0, 1.1, boxstyle="round,pad=0.03", fc="#fffbe6", ec="#b59b2a", zorder=3))
ax.text(3.4, 1.55, "Decision support only: the system never\napproves, rejects or denies a request.\nOnly staff / admin record the final decision.", fontsize=8, va="top", zorder=4)
ax.add_patch(FancyBboxPatch((7.6, 0.55), 4.2, 1.1, boxstyle="round,pad=0.03", fc="#fffbe6", ec="#b59b2a", zorder=3))
ax.text(7.75, 1.55, "Prediction INPUTS: service, priority, queue size,\noffice, submission weekday.\nEXCLUDED (leakage): Processing_Days, Status, Reopened", fontsize=7.4, va="top", zorder=4)
ax.add_patch(Ellipse((12.45, 8.8), 0.4, 0.22, fc="#e8f1fb", ec="#1f3a5f")); ax.text(12.75, 8.8, "Midterm workflow", fontsize=7.5, va="center")
ax.add_patch(Ellipse((12.45, 8.45), 0.4, 0.22, fc="#fff1d6", ec="#1f3a5f")); ax.text(12.75, 8.45, "New: analytics / deployment", fontsize=7.5, va="center")
plt.savefig("assets/use_case_diagram.png", dpi=170, bbox_inches="tight", facecolor="white"); plt.close()

# ------------------------------------------------------------------ workflow
fig, ax = plt.subplots(figsize=(15, 11)); ax.set_xlim(0, 15); ax.set_ylim(0, 11); ax.axis("off")
ax.text(7.5, 10.7, "Analytical Process Flow and Human Decision Workflow", ha="center", fontsize=14, fontweight="bold")
def box(x, y, w, h, text, fc="#e8f1fb", ec="#1f3a5f", fs=8.8, bold=False):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.04", fc=fc, ec=ec, lw=1.4, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=4, fontweight="bold" if bold else "normal")
def arrow(a, b, label=None, lo=(0, 0.14), color="#333"):
    ax.annotate("", xy=b, xytext=a, zorder=2, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, shrinkA=0, shrinkB=0))
    if label: ax.text((a[0]+b[0])/2+lo[0], (a[1]+b[1])/2+lo[1], label, fontsize=8, ha="center", style="italic", color=color,
                      bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9), zorder=5)
ax.text(0.3, 10.2, "A. Analytical process flow (offline, reused from the midterm)", fontsize=10.5, fontweight="bold", color="#1f3a5f")
xs = [1.35, 3.6, 5.85, 8.1, 10.35, 12.6]
labs = ["Upload CSV\n(Dataset page)", "Validate schema\nnulls / duplicates", "Clean + engineer\nfeatures\n(weekday, queue cat.)",
        "EDA + charts\n(Data Analysis)", "Train once:\nDecision Tree d=3\nsave .joblib", "Saved pipeline\nloaded & cached\n(Prediction page)"]
for x, l in zip(xs, labs): box(x, 9.0, 1.95, 1.25, l, fc="#fff1d6" if "Saved" in l or "Train" in l else "#e8f1fb")
for a, b in zip(xs[:-1], xs[1:]): arrow((a+0.98, 9.0), (b-0.98, 9.0))

ax.text(0.3, 7.85, "B. Decision workflow (online, per request)", fontsize=10.5, fontweight="bold", color="#1f3a5f")
# lanes
for (y0, h, name, c) in [(5.95, 1.7, "Student", "#f3f6fb"), (3.85, 1.95, "System (model)", "#fdf7e8"), (0.35, 3.35, "Staff / Administrator", "#eef7ee")]:
    ax.add_patch(Rectangle((0.3, y0), 14.4, h, fc=c, ec="#b0b7c3", lw=1, zorder=0))
    ax.text(14.6, y0+h-0.1, name, fontsize=9, fontweight="bold", va="top", ha="right", color="#445")
box(2.3, 6.8, 2.7, 0.9, "Student submits\nservice request")
box(2.3, 4.85, 2.9, 1.1, "Extract 5 submission-time\nfeatures only\n(no Processing_Days/Status)")
box(5.8, 4.85, 2.9, 1.1, "Saved model predicts\nbreach risk + probability\n(+ version, timestamp)", fc="#fff1d6")
# diamond
dx, dy = 9.1, 4.85
ax.add_patch(Polygon([(dx-1.2, dy), (dx, dy+0.75), (dx+1.2, dy), (dx, dy-0.75)], fc="#fff1d6", ec="#1f3a5f", lw=1.4, zorder=3))
ax.text(dx, dy, "Risk >= 50%?", ha="center", va="center", fontsize=8.8, zorder=4)
box(12.6, 4.85, 3.0, 1.0, "Normal processing\n(no flag, still monitored)")
box(9.1, 2.95, 3.0, 0.95, "Added to review queue\n(Dashboard)", fc="#e1f2e1", ec="#2e7d32")
box(5.8, 2.95, 3.0, 0.95, "Staff reviews context:\nqueue, office, requester, risk", fc="#e1f2e1", ec="#2e7d32")
box(2.4, 2.95, 3.2, 1.3, "Final HUMAN decision:\nexpedite | reassign | contact\nrequester | monitor | override", fc="#e1f2e1", ec="#2e7d32", bold=True)
box(2.4, 1.15, 3.4, 0.95, "Decision logged: reviewer, note,\nmodel version, timestamp", fc="#e1f2e1", ec="#2e7d32")
box(6.6, 1.15, 3.2, 0.95, "Request completed:\nactual SLA outcome recorded", fc="#e1f2e1", ec="#2e7d32")
box(11.2, 1.15, 3.8, 0.95, "Append to dataset -> monthly\nretrain & compare metrics", fc="#e1f2e1", ec="#2e7d32")
arrow((2.3, 6.35), (2.3, 5.42)); arrow((3.78, 4.85), (4.33, 4.85)); arrow((7.28, 4.85), (7.9, 4.85))
arrow((10.3, 4.85), (11.1, 4.85), "No", lo=(0, 0.15)); arrow((9.1, 4.1), (9.1, 3.43), "Yes", lo=(0.3, 0))
arrow((7.6, 2.95), (7.32, 2.95)); arrow((4.3, 2.95), (4.02, 2.95)); 
arrow((2.4, 2.3), (2.4, 1.63)); arrow((4.1, 1.15), (5.0, 1.15)); arrow((8.2, 1.15), (9.3, 1.15))
arrow((12.6, 4.35), (12.6, 1.63), "request still completes;\noutcome recorded", lo=(0.0, 0.0))
plt.savefig("assets/decision_workflow.png", dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
