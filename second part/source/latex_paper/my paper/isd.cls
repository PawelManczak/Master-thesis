\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{isd}[]
\LoadClass[11pt,twoside]{article}

\RequirePackage{titlesec}

\RequirePackage{ifthen}

\RequirePackage{calc}

\RequirePackage[a4paper,
                left=3.17cm,right=3.17cm,top=1.27cm,bottom=2.18cm,
                includehead,includefoot,headsep=0.8cm,footskip=8mm,headheight=11pt]{geometry}

\RequirePackage{times}
\RequirePackage{xparse}

\RequirePackage{fancyhdr}

\RequirePackage{verbatim}

\RequirePackage{xstring}

\RequirePackage{pgffor}

\RequirePackage{caption}

% \RequirePackage[numbers]{natbib}


%%%%%%%%%% FORMATIRANJE NASLOVA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\setcounter{secnumdepth}{2}

\titleformat{\section}
  {\fontsize{12pt}{16pt}\selectfont\bfseries}{\thesection.}{0.4cm}{}

\titleformat{\subsection}
  {\fontsize{11pt}{15pt}\selectfont\bfseries}{\thesubsection.}{0.4cm}{}

\titleformat{\subsubsection}
  {\fontsize{11pt}{15pt}\selectfont\bfseries\itshape}{}{0.4cm}{}

\titleformat{\paragraph}
  {\fontsize{11pt}{15pt}\selectfont\itshape}{}{0pt}{}

\titlespacing*{\section}{0pt}{15pt}{5pt}
\titlespacing*{\subsection}{0pt}{15pt}{5pt}
\titlespacing*{\paragraph}{0pt}{15pt}{5pt}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%  PARAGRAF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\setlength{\parskip}{0pt}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%% REFERENCE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% \def\@biblabel#1{\hspace*{0.4cm}#1.\hspace*{2mm}}
% \setlength{\bibsep}{0pt}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%% OPISI SLIKA I TABLICA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\captionsetup{
   labelfont={bf,small},
   textfont=small,
   labelsep=period,
   indention=20pt,
   justification=justified,
   margin=10pt,
   skip=0.64cm
}

\captionsetup[figure]{name={Fig.}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%% FOOTNOTE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\setlength{\skip\footins}{0.6cm}%razmak izmedju dna teksta i footnote
\setlength{\footskip}{1cm}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%% HEADERI I FOOTERI OD PRVE I PREOSTALIH STRANICA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\fancypagestyle{firstpage}{
           \fancyhf{}
           \chead{\fontsize{8}{10}\selectfont \textsc{34\textsuperscript{RD} International Conference On Information Systems Development (ISD2026 Prague, Czechia)}}
           }

\fancypagestyle{followingpage}{
           \fancyhf{}
           \fancyhead[LE]{\fontsize{8}{10}\selectfont \textsc{\showautori{}}}
           \fancyhead[RO]{\fontsize{8}{10}\selectfont \textsc{ISD2026 Prague, Czechia}}
           \fancyhead[RE]{\fontsize{8}{10}\selectfont \textsc{\showshorttitle{}}}
           }

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%% KRATKI NASLOV %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\let\@shorttitle\empty
\def\shorttitle#1{\gdef\@shorttitle{#1}}


\newcommand{\showshorttitle}{
\ifthenelse{\equal{\@shorttitle}{}}%
{
\@title\relax}%
{
\@shorttitle\relax%
}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%% AUTORI %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\let\@autori\empty
\def\autori#1{\gdef\@autori{#1}}


\let\@shortautori\empty
\def\shortautori#1{\gdef\@shortautori{#1}}


\let\@email\empty
\def\email#1{\gdef\@email{#1}}

\let\@institution\empty
\def\institution#1{\gdef\@institution{#1}}

\let\@city\empty
\def\city#1{\gdef\@city{#1}}

\let\@country\empty
\def\country#1{\gdef\@country{#1}}


\newcommand{\showautori}{
\ifthenelse{\equal{\@shortautori}{}}%
{
\@autori\relax}%
{
\@shortautori\relax%
}
}


\newcommand*{\pojediniAutor}[2]{
  \StrBetween[{#1},{\numexpr(#1)+1\relax}]{,#2,}{,}{,}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


\renewcommand{\maketitle}{
             \noindent\mbox{\parbox{\linewidth-2mm}{\vspace*{1mm}
             \begin{centriraj}\fontsize{14pt}{18pt}\selectfont\bfseries\@title\end{centriraj}\vspace*{8mm}}}
             \StrCount{\@autori,}{,}[\broj]\noindent\\
             \foreach \n in {1,...,\broj}{\noindent
              \fontsize{12}{16}\selectfont\textbf{\textit{\pojediniAutor{\n}{\@autori}}}\\
              \fontsize{12}{16}\selectfont\textit{\pojediniAutor{\n}{\@institution}}\\
              \fontsize{12}{16}\selectfont\textit{\pojediniAutor{\n}{\@city}},\textit{\pojediniAutor{\n}{\@country}}\hfill\fontsize{12}{16}\selectfont\textit{\pojediniAutor{\n}{\@email}}\vspace*{4mm}\\
             }
}


\newenvironment{centriraj}{
  \setlength\topsep{0cm}
  \setlength\parskip{0pt}
  \begin{center}
}{
  \end{center}}

\renewenvironment{abstract}
     {
     \begin{list}{}{\setlength{\leftmargin}{0.64cm}\setlength{\rightmargin}{0.64cm}\setlength{\topsep}{0pt}\setlength{\parsep}{0pt}\setlength{\itemsep}{5pt}}
           \item[]\begin{centriraj}\fontsize{11pt}{15pt}\selectfont\bfseries Abstract\end{centriraj}
           \item[]\fontsize{10pt}{13pt}\selectfont }
     { \end{list} }


\pagestyle{followingpage}

\AtBeginDocument{\thispagestyle{firstpage}}


\ProcessOptions*






\DeclareDocumentCommand{\ISDtitle}{O{} m}{
% #1 je ono kaj hocemo za shorttitle
% #2 je pravi title
\gdef\@title{#2}%
\IfNoValueTF{#1}
{
\shorttitle{#2}%
}
{
\shorttitle{#1}%
}
}


\DeclareDocumentCommand{\ISDautori}{O{} m}{
% #1 skracena imena autora
% #2 puna imena
\gdef\@autori{#2}
\IfNoValueTF{#1}
{
\autori{#2}%
}
{
\gdef\@shortautori{#1}
\shortautori{#1}%
}
}


\let\title\ISDtitle
\let\author\ISDautori
