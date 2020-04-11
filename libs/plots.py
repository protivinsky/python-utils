import os
from yattag import Doc, indent
from libs.utils import create_stamped_temp, slugify
import matplotlib.pyplot as plt

# NOTE - Does not work out of the box, needs a fix:
#
# Annoyingly, the js loading of subpages violates Cross-Origin Requests policy in all browsers
# when files are served locally via file:///. Works fine for http protocol though.
# It is possible to use iframes rather than js loader, but it's ugly and has other issues (multiple nested scrollbars).
#
# Workarounds:
#   - Firefox:
#       - go to about:config -> search for privacy.file_unique_origin and toggle
#       - then set up Firefox as the default for opening .htm files (that's the reason why I do not use .html)
#   - Chrome
#       - can be started with "--allow-file-access-from-files", then it should just work
#       - it would be possible to start the appropriate process in .show, but I have not tried
#           - one workaround is enough for me
#       - https://stackoverflow.com/a/18137280
#   - Edge:
#       - until recently, it was the only browser not enforcing the CORS policy for local files, so it just
#           worked. The new version of Edge enforces the same, do not know how to get around there.
#   - or it is possible to use local webserver and serve the files via it
#       - CORS policy is respected with http
#       - python webserver works fine, just serving the directory: python -m http.server 8000
#       - however seems more hassle than just changing firefox config...


class Chart:

    def __init__(self, figs, cols=3, title=None, format='png'):
        if not isinstance(figs, list):
            figs = [figs]
        self.figs = [f if isinstance(f, plt.Figure) else f.get_figure() for f in figs]
        self.cols = cols
        self.format = format
        self.title = title or self.figs[0].axes[0].title._text

    def save(self, path, inner=False):
        os.makedirs(path, exist_ok=True)
        n = len(self.figs)
        for i in range(n):
            self.figs[i].savefig(f'{path}/fig_{i+1:03d}.{self.format}')
        plt.close('all')

        doc, tag, text = Doc().tagtext()

        doc.asis('<!DOCTYPE html>')
        with tag('html'):
            with tag('head'):
                with tag('title'):
                    text(self.title or 'Chart')
            with tag('body'):
                with tag('h1'):
                    text(self.title or 'Chart')
                num_rows = (n + self.cols - 1) // self.cols
                for r in range(num_rows):
                    with tag('div'):
                        for c in range(min(self.cols, n - self.cols * r)):
                            doc.stag('img', src=f'fig_{self.cols * r + c + 1:03d}.{self.format}')

        file = open('{}/page.htm'.format(path), 'w', encoding='utf-8')
        file.write(indent(doc.getvalue()))
        file.close()

    def show(self):
        path = create_stamped_temp('reports')
        self.save(path)
        os.startfile('{}/page.htm'.format(path))


# I am not using it at the end, not sure if it works correctly.
class Text:

    def __init__(self, texts, width=750, title=None):
        if not isinstance(texts, list):
            texts = [texts]
        self.texts = texts
        self.width = width
        self.title = title

    def save(self, path, inner=False):
        os.makedirs(path, exist_ok=True)

        doc, tag, text = Doc().tagtext()

        doc.asis('<!DOCTYPE html>')
        with tag('html'):
            with tag('head'):
                with tag('title'):
                    text(self.title or 'Text')
            with tag('body'):
                with tag('h1'):
                    text(self.title or 'Text')
                with tag('div'):
                    for t in self.texts:
                        with tag('div', style='width: {}px; float: left'.format(self.width)):
                            with tag('pre'):
                                text(t)

        file = open('{}/page.htm'.format(path), 'w', encoding='utf-8')
        file.write(indent(doc.getvalue()))
        file.close()

    def show(self):
        path = create_stamped_temp('reports')
        self.save(path)
        os.startfile('{}/page.htm'.format(path))


class Selector:

    def __init__(self, charts, title=None):
        if not isinstance(charts, list):
            charts = [charts]
        self.charts = [ch if isinstance(ch, (Text, Chart, Selector)) else Chart(ch) for ch in charts]
        self.title = title or 'Selector'

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        n = len(self.charts)
        for i in range(n):
            ch = self.charts[i]
            if ch.title is None:
                ch.title = '{}_{:02d}'.format('Chart' if isinstance(ch, Chart) else ('Text' if isinstance(ch, Text)
                    else 'Selector'), i)
            ch.save('{}/{}'.format(path, slugify(ch.title)))

        doc, tag, text, line = Doc().ttl()

        doc.asis('<!DOCTYPE html>')
        with tag('html'):
            with tag('head'):
                with tag('title'):
                    text(self.title or 'Selector')
                with tag('script'):
                    doc.asis("""
      function loader(target, file) {
        var element = document.getElementById(target);
        var xmlhttp = new XMLHttpRequest();
        xmlhttp.onreadystatechange = function(){
          if(xmlhttp.status == 200 && xmlhttp.readyState == 4){          
            var txt = xmlhttp.responseText;
            var next_file = ""
            var matches = txt.match(/<script>loader\\('.*', '(.*)'\\)<\\/script>/);
            if (matches) {
              next_file = matches[1];
            };            
            txt = txt.replace(/^[\s\S]*<body>/, "").replace(/<\/body>[\s\S]*$/, "");
            txt = txt.replace(/src=\\"fig_/g, "src=\\"" + file + "/fig_");
            txt = txt.replace(/loader\\('/g, "loader('" + file.replace("/", "-") + "-");
            txt = txt.replace(/div id=\\"/, "div id=\\"" + file.replace("/", "-") + "-");
            txt = txt.replace(/content', '/g, "content', '" + file + "/");
            element.innerHTML = txt;
            if (next_file) {
              loader(file.replace("/", "-") + "-content", file.replace("/", "-") + "/" + next_file);
            };            
          };
        };
        xmlhttp.open("GET", file + "/page.htm", true);
        xmlhttp.send();
      }
    """)
            with tag('body'):
                with tag('h1'):
                    text(self.title or 'Selector')
                with tag('div'):
                    for ch in self.charts:
                        #line('a', ch.title, href='{}/page.html'.format(slugify(ch.title)), target='iframe')
                        line('button', ch.title, type='button',
                             onclick='loader(\'content\', \'{}\')'.format(slugify(ch.title)))
                with tag('div', id='content'):
                    text('')
                with tag('script'):
                    doc.asis('loader(\'content\', \'{}\')'.format(slugify(self.charts[0].title)))

        file = open('{}/page.htm'.format(path), 'w', encoding='utf-8')
        file.write(indent(doc.getvalue()))
        file.close()

    def show(self):
        path = create_stamped_temp('reports')
        self.save(path)
        os.startfile('{}/page.htm'.format(path))


