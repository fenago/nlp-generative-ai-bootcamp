
# Creativity with AI: Creating Images with DALL·E APIs in Google Colab {#13ec .pw-post-title .fp .fq .fr .be .fs .ft .fu .fv .fw .fx .fy .fz .ga .gb .gc .gd .ge .gf .gg .gh .gi .gj .gk .gl .gm .gn .go .gp .gq .gr .bj testid="storyTitle" selectable-paragraph=""}

::: {.gs .gt .gu .gv .gw}
::: {.speechify-ignore .ab .co}
::: {.speechify-ignore .bg .l}
::: {.gx .gy .gz .ha .hb .ab}
<div>

::: {.ab .hc}
[](https://drlee.io/?source=post_page-----15d1b9859508--------------------------------){rel="noopener  ugc nofollow"}

<div>

::: {.bl aria-hidden="false" aria-describedby="1" aria-labelledby="1"}
::: {.l .hd .he .bx .hf .hg}
::: {.l .ee}
![Dr. Ernesto Lee](./Lab_3_files/1_nzdYUSs4c2RQs2W0FCHv1g.jpg){.l .eq
.bx .dc .dd .cw width="44" height="44" loading="lazy"
testid="authorPhoto"}

::: {.hh .bx .l .dc .dd .eo .n .hi .ep}
:::
:::
:::
:::

</div>
:::

</div>

::: {.bm .bg .l}
::: ab
::: {style="flex:1"}
[]{.be .b .bf .z .bj}

::: {.hj .ab .q}
::: {.ab .q .hk}
::: {.ab .q}
<div>

::: {.bl aria-hidden="false" aria-describedby="2" aria-labelledby="2"}
[Dr. Ernesto
Lee](https://drlee.io/?source=post_page-----15d1b9859508--------------------------------){.af
.ag .ah .ai .aj .ak .al .am .an .ao .ap .aq .ar .hn testid="authorName"
rel="noopener  ugc nofollow"}
:::

</div>
:::

::: {.ho .hp .l}
::: {.ab .hq}
::: ab
![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdib3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSI+PHBhdGggZD0iTTE1LjE2IDhjMCAuNjUtLjQ2IDEuMTQtLjg2IDEuNTctLjIzLjI1LS40Ny41LS41Ni43Mi0uMS4yMi0uMDkuNTUtLjEuODggMCAuNi0uMDEgMS4zLS40OCAxLjc4LS40OC40OC0xLjE2LjUtMS43NS41LS4zMiAwLS42NS4wMS0uODYuMS0uMi4wNy0uNDYuMzMtLjcuNTctLjQyLjQxLS45Ljg4LTEuNTQuODhzLTEuMTItLjQ3LTEuNTQtLjg4YTIuODcgMi44NyAwIDAgMC0uNy0uNThjLS4yMi0uMDktLjU0LS4wOC0uODctLjA5LS41OSAwLTEuMjctLjAyLTEuNzQtLjVzLS40OC0xLjE3LS40OS0xLjc4YzAtLjMzLS4wMS0uNjctLjEtLjg4LS4wNy0uMi0uMzItLjQ3LS41NS0uNzEtLjQtLjQ0LS44Ny0uOTMtLjg3LTEuNThzLjQ2LTEuMTQuODctMS41OGMuMjMtLjI0LjQ3LS41LjU2LS43MS4wOS0uMjIuMDgtLjU1LjA5LS44OCAwLS42LjAyLTEuMy40OS0xLjc4czEuMTUtLjUgMS43NC0uNWMuMzMgMCAuNjYtLjAxLjg2LS4xLjItLjA4LjQ3LS4zMy43LS41Ny40My0uNDEuOTEtLjg4IDEuNTUtLjg4LjYzIDAgMS4xMi40NyAxLjU0Ljg4LjI0LjI0LjQ5LjQ4LjcuNTguMjIuMDkuNTQuMDguODYuMDkuNiAwIDEuMjcuMDIgMS43NS41LjQ3LjQ4LjQ4IDEuMTcuNDkgMS43OCAwIC4zMyAwIC42Ny4wOS44OC4wOC4yLjMzLjQ3LjU2LjcxLjQuNDQuODYuOTMuODYgMS41OHoiIGZpbGw9IiM0MzdBRkYiPjwvcGF0aD48cGF0aCBkPSJNNy4zMyAxMC41Yy4yIDAgLjM4LjA4LjUyLjIyLjEzLjE0LjIxLjMzLjIxLjUzIDAgLjA3LjAzLjEzLjA3LjE4YS4yNC4yNCAwIDAgMCAuMzUgMCAuMjUuMjUgMCAwIDAgLjA3LS4xOGMwLS4yLjA4LS4zOS4yMi0uNTNhLjczLjczIDAgMCAxIC41Mi0uMjJoMS45NmMuMTMgMCAuMjUtLjA1LjM0LS4xNWEuNS41IDAgMCAwIC4xNS0uMzVWNmEuNS41IDAgMCAwLS4xNS0uMzUuNDguNDggMCAwIDAtLjM0LS4xNUg5Ljc4Yy0uMzMgMC0uNjQuMTMtLjg3LjM3LS4yMy4yMy0uMzYuNTUtLjM2Ljg4djIuNWMwIC4wNy0uMDIuMTMtLjA3LjE4YS4yNC4yNCAwIDAgMS0uMzUgMCAuMjUuMjUgMCAwIDEtLjA3LS4xOHYtMi41YzAtLjMzLS4xMy0uNjUtLjM2LS44OGExLjIxIDEuMjEgMCAwIDAtLjg2LS4zN0g1LjM3YS40OC40OCAwIDAgMC0uMzUuMTUuNS41IDAgMCAwLS4xNC4zNXY0YzAgLjEzLjA1LjI2LjE0LjM1LjEuMS4yMi4xNS4zNS4xNWgxLjk2eiIgZmlsbD0iI2ZmZiI+PC9wYXRoPjwvc3ZnPg==)
:::
:::
:::

[[·]{.be .b .bf .z .dw}]{.hr .hs aria-hidden="true"}

Follow
:::
:::
:::
:::

::: {.l .hy}
[]{.be .b .bf .z .dw}

::: {.ab .cm .hz .ia .ib}
[]{.be .b .bf .z .dw}

::: {.ab .ae}
[5 min read]{testid="storyReadTime"}

::: {.ic .id .l aria-hidden="true"}
[[·]{.be .b .bf .z .dw}]{.l aria-hidden="true"}
:::

[Nov 12, 2023]{testid="storyPublishDate"}
:::
:::
:::
:::
:::

::: {.ab .co .ie .if .ig .ih .ii .ij .ik .il .im .in .io .ip .iq .ir .is .it}
::: {.h .k .w .eb .ec .q}
::: {.jj .l}
::: {.ab .q .jk}
::: {.pw-multi-vote-icon .ee .jl .jm .jn .jo}
<div>

<div>

::: {.bl aria-hidden="false" aria-describedby="74" aria-labelledby="74"}
![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdib3g9IjAgMCAyNCAyNCIgYXJpYS1sYWJlbD0iY2xhcCI+PHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0xMS4zNy44M0wxMiAzLjI4bC42My0yLjQ1aC0xLjI2ek0xMy45MiAzLjk1bDEuNTItMi4xLTEuMTgtLjQtLjM0IDIuNXpNOC41OSAxLjg0bDEuNTIgMi4xMS0uMzQtMi41LTEuMTguNHpNMTguNTIgMTguOTJhNC4yMyA0LjIzIDAgMCAxLTIuNjIgMS4zM2wuNDEtLjM3YzIuMzktMi40IDIuODYtNC45NSAxLjQtNy42M2wtLjkxLTEuNi0uOC0xLjY3Yy0uMjUtLjU2LS4xOS0uOTguMjEtMS4yOWEuNy43IDAgMCAxIC41NS0uMTNjLjI4LjA1LjU0LjIzLjcyLjVsMi4zNyA0LjE2Yy45NyAxLjYyIDEuMTQgNC4yMy0xLjMzIDYuN3ptLTExLS40NGwtNC4xNS00LjE1YS44My44MyAwIDAgMSAxLjE3LTEuMTdsMi4xNiAyLjE2YS4zNy4zNyAwIDAgMCAuNTEtLjUybC0yLjE1LTIuMTZMMy42IDExLjJhLjgzLjgzIDAgMCAxIDEuMTctMS4xN2wzLjQzIDMuNDRhLjM2LjM2IDAgMCAwIC41MiAwIC4zNi4zNiAwIDAgMCAwLS41Mkw1LjI5IDkuNTFsLS45Ny0uOTdhLjgzLjgzIDAgMCAxIDAtMS4xNi44NC44NCAwIDAgMSAxLjE3IDBsLjk3Ljk3IDMuNDQgMy40M2EuMzYuMzYgMCAwIDAgLjUxIDAgLjM3LjM3IDAgMCAwIDAtLjUyTDYuOTggNy44M2EuODIuODIgMCAwIDEtLjE4LS45LjgyLjgyIDAgMCAxIC43Ni0uNTFjLjIyIDAgLjQzLjA5LjU4LjI0bDUuOCA1Ljc5YS4zNy4zNyAwIDAgMCAuNTgtLjQyTDEzLjQgOS42N2MtLjI2LS41Ni0uMi0uOTguMi0xLjI5YS43LjcgMCAwIDEgLjU1LS4xM2MuMjguMDUuNTUuMjMuNzMuNWwyLjIgMy44NmMxLjMgMi4zOC44NyA0LjU5LTEuMjkgNi43NWE0LjY1IDQuNjUgMCAwIDEtNC4xOSAxLjM3IDcuNzMgNy43MyAwIDAgMS00LjA3LTIuMjV6bTMuMjMtMTIuNWwyLjEyIDIuMTFjLS40MS41LS40NyAxLjE3LS4xMyAxLjlsLjIyLjQ2LTMuNTItMy41M2EuODEuODEgMCAwIDEtLjEtLjM2YzAtLjIzLjA5LS40My4yNC0uNTlhLjg1Ljg1IDAgMCAxIDEuMTcgMHptNy4zNiAxLjdhMS44NiAxLjg2IDAgMCAwLTEuMjMtLjg0IDEuNDQgMS40NCAwIDAgMC0xLjEyLjI3Yy0uMy4yNC0uNS41NS0uNTguODktLjI1LS4yNS0uNTctLjQtLjkxLS40Ny0uMjgtLjA0LS41NiAwLS44Mi4xbC0yLjE4LTIuMThhMS41NiAxLjU2IDAgMCAwLTIuMiAwYy0uMi4yLS4zMy40NC0uNC43YTEuNTYgMS41NiAwIDAgMC0yLjYzLjc1IDEuNiAxLjYgMCAwIDAtMi4yMy0uMDQgMS41NiAxLjU2IDAgMCAwIDAgMi4yYy0uMjQuMS0uNS4yNC0uNzIuNDVhMS41NiAxLjU2IDAgMCAwIDAgMi4ybC41Mi41MmExLjU2IDEuNTYgMCAwIDAtLjc1IDIuNjFMNyAxOWE4LjQ2IDguNDYgMCAwIDAgNC40OCAyLjQ1IDUuMTggNS4xOCAwIDAgMCAzLjM2LS41IDQuODkgNC44OSAwIDAgMCA0LjItMS41MWMyLjc1LTIuNzcgMi41NC01Ljc0IDEuNDMtNy41OUwxOC4xIDcuNjh6Ij48L3BhdGg+PC9zdmc+)
:::

</div>

</div>
:::

::: {.pw-multi-vote-count .l .jz .ka .kb .kc .kd .ke .kf}
<div>

::: {.bl aria-hidden="false" aria-describedby="75" aria-labelledby="75"}
60[]{.l .h .g .f .qb .qc}
:::

</div>
:::
:::
:::

<div>

::: {.bl aria-hidden="false" aria-describedby="3" aria-labelledby="3"}
![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdib3g9IjAgMCAyNCAyNCIgY2xhc3M9ImtrIj48cGF0aCBkPSJNMTggMTYuOGE3LjE0IDcuMTQgMCAwIDAgMi4yNC01LjMyYzAtNC4xMi0zLjUzLTcuNDgtOC4wNS03LjQ4QzcuNjcgNCA0IDcuMzYgNCAxMS40OGMwIDQuMTMgMy42NyA3LjQ4IDguMiA3LjQ4YTguOSA4LjkgMCAwIDAgMi4zOC0uMzJjLjIzLjIuNDguMzkuNzUuNTYgMS4wNi42OSAyLjIgMS4wNCAzLjQgMS4wNC4yMiAwIC40LS4xMS40OC0uMjlhLjUuNSAwIDAgMC0uMDQtLjUyIDYuNCA2LjQgMCAwIDEtMS4xNi0yLjY1di4wMnptLTMuMTIgMS4wNmwtLjA2LS4yMi0uMzIuMWE4IDggMCAwIDEtMi4zLjMzYy00LjAzIDAtNy4zLTIuOTYtNy4zLTYuNTlTOC4xNyA0LjkgMTIuMiA0LjljNCAwIDcuMSAyLjk2IDcuMSA2LjYgMCAxLjgtLjYgMy40Ny0yLjAyIDQuNzJsLS4yLjE2di4yNmwuMDIuM2E2Ljc0IDYuNzQgMCAwIDAgLjg4IDIuNCA1LjI3IDUuMjcgMCAwIDEtMi4xNy0uODZjLS4yOC0uMTctLjcyLS4zOC0uOTQtLjU5bC4wMS0uMDJ6Ij48L3BhdGg+PC9zdmc+){.kk}
:::

</div>
:::

::: {.ab .q .iu .iv .iw .ix .iy .iz .ja .jb .jc .jd .je .jf .jg .jh .ji}
::: {.kl .k .j .i .d}
:::

::: {.h .k}
<div>

::: {.bl aria-hidden="false" aria-describedby="4" aria-labelledby="4"}
::: {.bl aria-hidden="false"}
![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdib3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgY2xhc3M9ImtxIj48cGF0aCBkPSJNMTcuNSAxLjI1YS41LjUgMCAwIDEgMSAwdjIuNUgyMWEuNS41IDAgMCAxIDAgMWgtMi41djIuNWEuNS41IDAgMCAxLTEgMHYtMi41SDE1YS41LjUgMCAwIDEgMC0xaDIuNXYtMi41em0tMTEgNC41YTEgMSAwIDAgMSAxLTFIMTFhLjUuNSAwIDAgMCAwLTFINy41YTIgMiAwIDAgMC0yIDJ2MTRhLjUuNSAwIDAgMCAuOC40bDUuNy00LjQgNS43IDQuNGEuNS41IDAgMCAwIC44LS40di04LjVhLjUuNSAwIDAgMC0xIDB2Ny40OGwtNS4yLTRhLjUuNSAwIDAgMC0uNiAwbC01LjIgNFY1Ljc1eiIgZmlsbD0iIzAwMCI+PC9wYXRoPjwvc3ZnPg==){.kq}
:::
:::

</div>
:::

::: {.eq .kr .cm}
::: {.l .ae}
::: {.ab .ca}
::: {.ks .kt .ku .kv .kw .kx .ch .bg}
::: ab
<div>

[](https://medium.com/plans?dimension=post_audio_button&postId=15d1b9859508&source=upgrade_membership---post_audio_button----------------------------------){.af
.ag .ah .ai .aj .ak .al .am .an .ao .ap .aq .ar .as .at
rel="noopener follow"}

<div>

::: {.bl aria-hidden="false" aria-describedby="18" aria-labelledby="18"}
![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdib3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSI+PHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0zIDEyYTkgOSAwIDEgMSAxOCAwIDkgOSAwIDAgMS0xOCAwem05LTEwYTEwIDEwIDAgMSAwIDAgMjAgMTAgMTAgMCAwIDAgMC0yMHptMy4zOCAxMC40MmwtNC42IDMuMDZhLjUuNSAwIDAgMS0uNzgtLjQxVjguOTNjMC0uNC40NS0uNjMuNzgtLjQxbDQuNiAzLjA2Yy4zLjIuMy42NCAwIC44NHoiIGZpbGw9ImN1cnJlbnRDb2xvciI+PC9wYXRoPjwvc3ZnPg==)

::: {.j .i .d}
Listen
:::
:::

</div>

</div>
:::
:::
:::
:::
:::

::: {.bl aria-hidden="false" aria-describedby="postFooterSocialMenu" aria-labelledby="postFooterSocialMenu"}
<div>

::: {.bl aria-hidden="false" aria-describedby="6" aria-labelledby="6"}
![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdib3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSI+PHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0xNS4yMiA0LjkzYS40Mi40MiAwIDAgMS0uMTIuMTNoLjAxYS40NS40NSAwIDAgMS0uMjkuMDguNTIuNTIgMCAwIDEtLjMtLjEzTDEyLjUgM3Y3LjA3YS41LjUgMCAwIDEtLjUuNS41LjUgMCAwIDEtLjUtLjVWMy4wMmwtMiAyYS40NS40NSAwIDAgMS0uNTcuMDRoLS4wMmEuNC40IDAgMCAxLS4xNi0uMy40LjQgMCAwIDEgLjEtLjMybDIuOC0yLjhhLjUuNSAwIDAgMSAuNyAwbDIuOCAyLjhhLjQyLjQyIDAgMCAxIC4wNy41em0tLjEuMTR6bS44OCAyaDEuNWEyIDIgMCAwIDEgMiAydjEwYTIgMiAwIDAgMS0yIDJoLTExYTIgMiAwIDAgMS0yLTJ2LTEwYTIgMiAwIDAgMSAyLTJIOGEuNS41IDAgMCAxIC4zNS4xNGMuMS4xLjE1LjIyLjE1LjM1YS41LjUgMCAwIDEtLjE1LjM1LjUuNSAwIDAgMS0uMzUuMTVINi40Yy0uNSAwLS45LjQtLjkuOXYxMC4yYS45LjkgMCAwIDAgLjkuOWgxMS4yYy41IDAgLjktLjQuOS0uOVY4Ljk2YzAtLjUtLjQtLjktLjktLjlIMTZhLjUuNSAwIDAgMSAwLTF6IiBmaWxsPSJjdXJyZW50Q29sb3IiPjwvcGF0aD48L3N2Zz4=)

::: {.j .i .d}
Share
:::
:::

</div>
:::

::: {.bl aria-hidden="false"}
::: {.bl aria-hidden="false"}
<div>

::: {.bl aria-hidden="false" aria-describedby="80" aria-labelledby="80"}
![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdib3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSI+PHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik00LjM5IDEyYzAgLjU1LjIgMS4wMi41OSAxLjQxLjM5LjQuODYuNTkgMS40LjU5LjU2IDAgMS4wMy0uMiAxLjQyLS41OS40LS4zOS41OS0uODYuNTktMS40MSAwLS41NS0uMi0xLjAyLS42LTEuNDFBMS45MyAxLjkzIDAgMCAwIDYuNCAxMGMtLjU1IDAtMS4wMi4yLTEuNDEuNTktLjQuMzktLjYuODYtLjYgMS40MXpNMTAgMTJjMCAuNTUuMiAxLjAyLjU4IDEuNDEuNC40Ljg3LjU5IDEuNDIuNTkuNTQgMCAxLjAyLS4yIDEuNC0uNTkuNC0uMzkuNi0uODYuNi0xLjQxIDAtLjU1LS4yLTEuMDItLjYtMS40MWExLjkzIDEuOTMgMCAwIDAtMS40LS41OWMtLjU1IDAtMS4wNC4yLTEuNDIuNTktLjQuMzktLjU4Ljg2LS41OCAxLjQxem01LjYgMGMwIC41NS4yIDEuMDIuNTcgMS40MS40LjQuODguNTkgMS40My41OS41NyAwIDEuMDQtLjIgMS40My0uNTkuMzktLjM5LjU3LS44Ni41Ny0xLjQxIDAtLjU1LS4yLTEuMDItLjU3LTEuNDFBMS45MyAxLjkzIDAgMCAwIDE3LjYgMTBjLS41NSAwLTEuMDQuMi0xLjQzLjU5LS4zOC4zOS0uNTcuODYtLjU3IDEuNDF6IiBmaWxsPSJjdXJyZW50Q29sb3IiPjwvcGF0aD48L3N2Zz4=)

::: {.j .i .d}
More
:::
:::

</div>
:::
:::
:::
:::
:::
:::
:::

</div>

![](./Lab_3_files/1_8txPISBrcaXrpUFAl4w95w.jpg){.bg .kx .mc .c
width="700" height="776" loading="eager" role="presentation"}

Welcome to the mesmerizing world of AI-generated art! Today, we're going
to explore how we can use OpenAI's DALL·E APIs to create, edit, and
manipulate images directly from our Google Colab notebooks. We'll focus
on a Belgian Malinois named Daisy to illustrate our examples. Let's dive
into how we can integrate DALL·E's capabilities into our Python code.

# Setting Up the OpenAI API in Google Colab {#1732 .nb .nc .fr .be .nd .ne .nf .ng .nh .ni .nj .nk .nl .nm .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .bj selectable-paragraph=""}

First things first, we need to set up our environment to communicate
with OpenAI's API. This is how you can do it in Google Colab:

``` {.oe .of .og .oh .oi .oj .ok .ol .bo .om .ba .bj}
!pip install openai
import os

# Prompt for the API key
api_key = input("Enter your OpenAI API key: ")

# Set the environment variable
os.environ["OPENAI_API_KEY"] = api_key
```

This snippet installs the OpenAI Python package and sets an environment
variable using the API key you provide. The API key is essential for
authentication when making requests to OpenAI's services.

# 1. Generate a New Image with DALL-E 3 {#e092 .nb .nc .fr .be .nd .ne .nf .ng .nh .ni .nj .nk .nl .nm .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .bj selectable-paragraph=""}

Now, let's generate an original image of Daisy, our Belgian Malinois.
We'll use a simple prompt and request the API to conjure up an image:

``` {.oe .of .og .oh .oi .oj .ok .ol .bo .om .ba .bj}
from openai import OpenAI
client = OpenAI()

response = client.images.generate(
  model="dall-e-3",
  prompt="a Belgian Malinois named Daisy sitting on the grass",
  size="1024x1024",
  quality="standard",
  n=1,
)

# To display the image in Google Colab, we'll use the following:
from IPython.display import Image
import requests

# Fetch the image from the URL
image_data = requests.get(response.data[0].url).content
display(Image(image_data))
```

![](./Lab_3_files/1_Rhzk4kE8dWwmASEjHrNzpg.png){.bg .kx .mc .c
width="700" height="700" loading="lazy" role="presentation"}

This code sends a request to generate an image of Daisy. We then fetch
the image from the provided URL and display it in the notebook.

# 2. Create a Mask for Inpainting {#a22d .nb .nc .fr .be .nd .ne .nf .ng .nh .ni .nj .nk .nl .nm .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .bj selectable-paragraph=""}

Inpainting is like digital magic --- it allows us to replace parts of an
image with new content. Let's say we want to place Daisy in front of the
Eiffel Tower. We would need a mask where Daisy is cut out. Here's how
you could do it in theory:

``` {.oe .of .og .oh .oi .oj .ok .ol .bo .om .ba .bj}
# This is a pseudo-code placeholder for creating a mask:
# mask = create_mask_function('Daisy.png')

# In a real scenario, you would use image editing software to create a mask
# where the areas to be replaced are transparent.

# Then you would upload it and use it in the API request
response = client.images.edit(
  model="dall-e-2",
  image=open("Daisy.png", "rb"),
  mask=open("mask.png", "rb"),
  prompt="a Belgian Malinois named Daisy in front of the Eiffel Tower",
  n=1,
  size="1024x1024"
)

# Displaying edited image in Google Colab
edited_image_data = requests.get(response.data[0].url).content
Image(BytesIO(edited_image_data))
```

# 3. Create a Variation {#54dc .nb .nc .fr .be .nd .ne .nf .ng .nh .ni .nj .nk .nl .nm .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .bj selectable-paragraph=""}

Maybe we want to see different versions of a sunflower in various
settings. This is where it starts:

![](./Lab_3_files/1_8txPISBrcaXrpUFAl4w95w.jpg){.bg .kx .mc .c
width="700" height="776" loading="lazy" role="presentation"}

To do this, we'll use the image variation endpoint:

``` {.oe .of .og .oh .oi .oj .ok .ol .bo .om .ba .bj}
from PIL import Image as PILImage
import io
import requests
from IPython.display import Image as IPyImage, display

# Load the JPEG image and ensure it's less than 4 MB as a PNG
with PILImage.open("/content/Sunflower_sky_backdrop.jpg") as img:
    img = img.convert("RGBA")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')

    # If the image is too large, reduce its size
    while len(img_byte_arr.getvalue()) > 4 * 1024 * 1024:
        img = img.resize((int(img.width * 0.75), int(img.height * 0.75)))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        if img.width < 200 or img.height < 200:  # Avoid making the image too small
            break

# Call the API to create image variations
response = client.images.create_variation(
  image=img_byte_arr.getvalue(),
  n=2,
  size="1024x1024"
)

# Displaying variations in Google Colab
for data in response.data:
    variation_image_data = requests.get(data.url).content
    display(IPyImage(variation_image_data))
```

![](./Lab_3_files/1_dQ8gnJB84GS7lMkHTmqsOQ.png){.bg .kx .mc .c
width="700" height="700" loading="lazy" role="presentation"}

![](./Lab_3_files/1_OWUmEUitjyOzK_ojBnPNzA.png){.bg .kx .mc .c
width="700" height="700" loading="lazy" role="presentation"}

These are the two variations that were created from the original.

# 4. Operating on Image Data {#1473 .nb .nc .fr .be .nd .ne .nf .ng .nh .ni .nj .nk .nl .nm .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .bj selectable-paragraph=""}

Before you send an image to DALL·E, you might want to resize or process
it. Here's how you can do that:

``` {.oe .of .og .oh .oi .oj .ok .ol .bo .om .ba .bj}
from PIL import Image as PILImage
from IPython.display import Image as IPyImage, display
import requests
from io import BytesIO

# Let's resize an image of Daisy
original_image = PILImage.open("/content/Sunflower_sky_backdrop.jpg")
new_size = (256, 256)
resized_image = original_image.resize(new_size)

resized_image_stream = BytesIO()
resized_image.save(resized_image_stream, format='PNG')

# Convert the BytesIO stream to bytes for the API call
resized_image_bytes = resized_image_stream.getvalue()

# Now, let's send this resized image to DALL·E
response = client.images.create_variation(
  image=resized_image_bytes,
  n=1,
  size="1024x1024"
)

# Displaying resized image in Google Colab
resized_image_data = requests.get(response.data[0].url).content
display(IPyImage(data=resized_image_data))
```

![](./Lab_3_files/1_UB4f83EYP0-UoVOx4qAWaA.png){.bg .kx .mc .c
width="700" height="423" loading="lazy" role="presentation"}

# 5. Error Handling {#f44d .nb .nc .fr .be .nd .ne .nf .ng .nh .ni .nj .nk .nl .nm .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .bj selectable-paragraph=""}

When dealing with API requests, we must be prepared for potential
errors. Here's how to handle them:

``` {.oe .of .og .oh .oi .oj .ok .ol .bo .om .ba .bj}
!pip install openai
from openai import OpenAIError
from PIL import Image
import io

# Convert the image to PNG and ensure it's less than 4 MB
with Image.open("/content/Sunflower_sky_backdrop.jpg") as img:
    img = img.convert("RGBA")  # Convert to RGBA to support transparency in PNG
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG', optimize=True)  # PNG format and optimization
    img_byte_arr.seek(0)  # Reset the stream position to the beginning

# Now attempt to create a variation with the converted image
try:
    response = client.images.create_variation(
      image=img_byte_arr,
      n=1,
      size="1024x1024"
    )
except OpenAIError as e:  # Corrected to catch OpenAIError directly
    print("An error occurred:", e.http_status)
    print("Details:", e.error)
```

This try-except block will catch any exceptions thrown by the API call
and print out details for debugging.
:::
:::
:::

::: {.ab .ca .ou .ov .ow .ox role="separator"}
[]{.oy .bx .bl .oz .pa .pb}[]{.oy .bx .bl .oz .pa .pb}[]{.oy .bx .bl .oz
.pa}
:::

::: {.fk .fl .fm .fn .fo}
::: {.ab .ca}
::: {.ch .bg .ew .ex .ey .ez}
With these examples, you now have a guide to generating and manipulating
images with DALL·E in Google Colab, all featuring Daisy, our Belgian
Malinois or a Sunflower. Remember, while the code provided here is for
illustrative purposes, actual implementation may require further tweaks,
especially when dealing with image files and API responses. Happy coding
and enjoy the creative journey with AI!
