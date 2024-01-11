

# Transforming Text into Lifelike Spoken English and Spanish Audio with OpenAI's Audio API in Google Colab {#8e43 .pw-post-title .fp .fq .fr .be .fs .ft .fu .fv .fw .fx .fy .fz .ga .gb .gc .gd .ge .gf .gg .gh .gi .gj .gk .gl .gm .gn .go .gp .gq .gr .bj testid="storyTitle" selectable-paragraph=""}

::: {.gs .gt .gu .gv .gw}
::: {.speechify-ignore .ab .co}
::: {.speechify-ignore .bg .l}
::: {.gx .gy .gz .ha .hb .ab}
<div>

::: {.ab .hc}
[](https://drlee.io/?source=post_page-----e35b0f7376b5--------------------------------){rel="noopener  ugc nofollow"}

<div>

::: {.bl aria-hidden="false" aria-describedby="1" aria-labelledby="1"}
::: {.l .hd .he .bx .hf .hg}
::: {.l .ee}
![Dr. Ernesto Lee](./Lab_1_files/1_nzdYUSs4c2RQs2W0FCHv1g.jpg){.l .eq
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
Lee](https://drlee.io/?source=post_page-----e35b0f7376b5--------------------------------){.af
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
[3 min read]{testid="storyReadTime"}

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

::: {.bl aria-hidden="false" aria-describedby="60" aria-labelledby="60"}
![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdib3g9IjAgMCAyNCAyNCIgYXJpYS1sYWJlbD0iY2xhcCI+PHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0xMS4zNy44M0wxMiAzLjI4bC42My0yLjQ1aC0xLjI2ek0xMy45MiAzLjk1bDEuNTItMi4xLTEuMTgtLjQtLjM0IDIuNXpNOC41OSAxLjg0bDEuNTIgMi4xMS0uMzQtMi41LTEuMTguNHpNMTguNTIgMTguOTJhNC4yMyA0LjIzIDAgMCAxLTIuNjIgMS4zM2wuNDEtLjM3YzIuMzktMi40IDIuODYtNC45NSAxLjQtNy42M2wtLjkxLTEuNi0uOC0xLjY3Yy0uMjUtLjU2LS4xOS0uOTguMjEtMS4yOWEuNy43IDAgMCAxIC41NS0uMTNjLjI4LjA1LjU0LjIzLjcyLjVsMi4zNyA0LjE2Yy45NyAxLjYyIDEuMTQgNC4yMy0xLjMzIDYuN3ptLTExLS40NGwtNC4xNS00LjE1YS44My44MyAwIDAgMSAxLjE3LTEuMTdsMi4xNiAyLjE2YS4zNy4zNyAwIDAgMCAuNTEtLjUybC0yLjE1LTIuMTZMMy42IDExLjJhLjgzLjgzIDAgMCAxIDEuMTctMS4xN2wzLjQzIDMuNDRhLjM2LjM2IDAgMCAwIC41MiAwIC4zNi4zNiAwIDAgMCAwLS41Mkw1LjI5IDkuNTFsLS45Ny0uOTdhLjgzLjgzIDAgMCAxIDAtMS4xNi44NC44NCAwIDAgMSAxLjE3IDBsLjk3Ljk3IDMuNDQgMy40M2EuMzYuMzYgMCAwIDAgLjUxIDAgLjM3LjM3IDAgMCAwIDAtLjUyTDYuOTggNy44M2EuODIuODIgMCAwIDEtLjE4LS45LjgyLjgyIDAgMCAxIC43Ni0uNTFjLjIyIDAgLjQzLjA5LjU4LjI0bDUuOCA1Ljc5YS4zNy4zNyAwIDAgMCAuNTgtLjQyTDEzLjQgOS42N2MtLjI2LS41Ni0uMi0uOTguMi0xLjI5YS43LjcgMCAwIDEgLjU1LS4xM2MuMjguMDUuNTUuMjMuNzMuNWwyLjIgMy44NmMxLjMgMi4zOC44NyA0LjU5LTEuMjkgNi43NWE0LjY1IDQuNjUgMCAwIDEtNC4xOSAxLjM3IDcuNzMgNy43MyAwIDAgMS00LjA3LTIuMjV6bTMuMjMtMTIuNWwyLjEyIDIuMTFjLS40MS41LS40NyAxLjE3LS4xMyAxLjlsLjIyLjQ2LTMuNTItMy41M2EuODEuODEgMCAwIDEtLjEtLjM2YzAtLjIzLjA5LS40My4yNC0uNTlhLjg1Ljg1IDAgMCAxIDEuMTcgMHptNy4zNiAxLjdhMS44NiAxLjg2IDAgMCAwLTEuMjMtLjg0IDEuNDQgMS40NCAwIDAgMC0xLjEyLjI3Yy0uMy4yNC0uNS41NS0uNTguODktLjI1LS4yNS0uNTctLjQtLjkxLS40Ny0uMjgtLjA0LS41NiAwLS44Mi4xbC0yLjE4LTIuMThhMS41NiAxLjU2IDAgMCAwLTIuMiAwYy0uMi4yLS4zMy40NC0uNC43YTEuNTYgMS41NiAwIDAgMC0yLjYzLjc1IDEuNiAxLjYgMCAwIDAtMi4yMy0uMDQgMS41NiAxLjU2IDAgMCAwIDAgMi4yYy0uMjQuMS0uNS4yNC0uNzIuNDVhMS41NiAxLjU2IDAgMCAwIDAgMi4ybC41Mi41MmExLjU2IDEuNTYgMCAwIDAtLjc1IDIuNjFMNyAxOWE4LjQ2IDguNDYgMCAwIDAgNC40OCAyLjQ1IDUuMTggNS4xOCAwIDAgMCAzLjM2LS41IDQuODkgNC44OSAwIDAgMCA0LjItMS41MWMyLjc1LTIuNzcgMi41NC01Ljc0IDEuNDMtNy41OUwxOC4xIDcuNjh6Ij48L3BhdGg+PC9zdmc+)
:::

</div>

</div>
:::

::: {.pw-multi-vote-count .l .jz .ka .kb .kc .kd .ke .kf}
<div>

::: {.bl aria-hidden="false" aria-describedby="61" aria-labelledby="61"}
77[]{.l .h .g .f .qd .qe}
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

[](https://medium.com/plans?dimension=post_audio_button&postId=e35b0f7376b5&source=upgrade_membership---post_audio_button----------------------------------){.af
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

::: {.bl aria-hidden="false" aria-describedby="66" aria-labelledby="66"}
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

![](./Lab_1_files/0_sAJRUFloZL0Ssvgq.jpg){.bg .kx .mc .c width="700"
height="700" loading="eager" role="presentation"}

here is a sample of the ChatGPT generated speech:

[https://github.com/fenago/datasets/raw/main/speech.mp3](https://github.com/fenago/datasets/raw/main/speech.mp3){.af
.nb rel="noopener ugc nofollow" target="_blank"}

# Introduction {#15fd .nc .nd .fr .be .ne .nf .ng .nh .ni .nj .nk .nl .nm .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .nz .bj selectable-paragraph=""}

In the digital age, the ability to convert text into spoken audio has
numerous applications, from making content more accessible to creating
engaging multimedia experiences. OpenAI's Audio API harnesses advanced
text-to-speech (TTS) technology, offering a user-friendly interface to
generate realistic voiceovers. With six unique voices and support for
multiple languages, this tool is ideal for various use cases, including
blog narration, multilingual audio production, and real-time streaming.

# Key Features {#202d .nc .nd .fr .be .ne .nf .ng .nh .ni .nj .nk .nl .nm .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .nz .bj selectable-paragraph=""}

-   [Six Distinct Voices: Alloy, Echo, Fable, Onyx, Nova,
    Shimmer.]{#efb7}
-   [Multiple Language Support: Follows Whisper model's language
    capabilities.]{#e9ff}
-   [Real-Time Audio Streaming: Utilizes chunk transfer
    encoding.]{#e0e7}
-   [Variety of Output Formats: MP3, Opus, AAC, FLAC.]{#a44f}

# Setting Up in Google Colab {#e5de .nc .nd .fr .be .ne .nf .ng .nh .ni .nj .nk .nl .nm .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .nz .bj selectable-paragraph=""}

Google Colab provides a powerful, cloud-based environment to run Python
code. Here, we will guide you through using OpenAI's Audio API in Colab.

# Initial Setup {#ad74 .nc .nd .fr .be .ne .nf .ng .nh .ni .nj .nk .nl .nm .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .nz .bj selectable-paragraph=""}

Start by importing the necessary module and setting up your OpenAI API
key:

``` {.on .oo .op .oq .or .os .ot .ou .bo .ov .ba .bj}
import os

# Prompt for the API key
api_key = input("Enter your OpenAI API key: ")

# Set the environment variable
os.environ["OPENAI_API_KEY"] = api_key
```

# Installation {#2e33 .nc .nd .fr .be .ne .nf .ng .nh .ni .nj .nk .nl .nm .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .nz .bj selectable-paragraph=""}

Upgrade to Python SDK v1.2 for compatibility with the Audio API:

``` {.on .oo .op .oq .or .os .ot .ou .bo .ov .ba .bj}
!pip install --upgrade openai
```

# Import Libraries {#8f1b .nc .nd .fr .be .ne .nf .ng .nh .ni .nj .nk .nl .nm .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .nz .bj selectable-paragraph=""}

Once the SDK is installed, import the required libraries, prep the data
and call the model:

``` {.on .oo .op .oq .or .os .ot .ou .bo .ov .ba .bj}
from pathlib import Path
from openai import OpenAI
client = OpenAI()
speech_file_path = Path("speech.mp3")
response = client.audio.speech.create(
  model="tts-1",
  voice="alloy",
  input="My name is Dr. Lee and I am a Data Analytics Professor at Miami Dade College.  I am trying this ChatGPT model for the first time and it is super cool!"
)
response.stream_to_file(speech_file_path)
from IPython.display import Audio

# Play the generated audio file
Audio("/content/speech.mp3")
```

# Explanation {#54c3 .nc .nd .fr .be .ne .nf .ng .nh .ni .nj .nk .nl .nm .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .nz .bj selectable-paragraph=""}

-   [`client.audio.speech.create()`{.cw .pb .pc .pd .ot .b}: Main
    function for speech generation.]{#87aa}
-   [`model`{.cw .pb .pc .pd .ot .b}: \"tts-1\" for standard quality,
    \"tts-1-hd\" for high definition.]{#0174}
-   [`voice`{.cw .pb .pc .pd .ot .b}: Choose from the six available
    options.]{#1ffb}
-   [`input`{.cw .pb .pc .pd .ot .b}: Your text to be converted into
    speech.]{#b9b1}
-   [`response.stream_to_file()`{.cw .pb .pc .pd .ot .b}: Saves the
    audio to the specified path.]{#d263}

# Use Cases and Applications {#510d .nc .nd .fr .be .ne .nf .ng .nh .ni .nj .nk .nl .nm .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .nz .bj selectable-paragraph=""}

# Narrating a Blog Post {#2f4e .nc .nd .fr .be .ne .nf .ng .nh .ni .nj .nk .nl .nm .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .nz .bj selectable-paragraph=""}

Transform your written content into an engaging audio format, making it
more accessible and appealing to a broader audience.

# Language Learning Tools {#64d3 .nc .nd .fr .be .ne .nf .ng .nh .ni .nj .nk .nl .nm .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .nz .bj selectable-paragraph=""}

Create multilingual educational content to aid in language learning.

# Real-Time Audio Streaming {#04f3 .nc .nd .fr .be .ne .nf .ng .nh .ni .nj .nk .nl .nm .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .nz .bj selectable-paragraph=""}

Develop interactive applications that require live voiceover, such as
virtual assistants or online gaming.

# Conclusion {#bd5c .nc .nd .fr .be .ne .nf .ng .nh .ni .nj .nk .nl .nm .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .nz .bj selectable-paragraph=""}

OpenAI's Audio API in Google Colab offers a versatile and powerful
platform for text-to-speech applications. Whether you're a content
creator, educator, or developer, this tool provides an accessible way to
enhance your projects with lifelike audio.

Remember to adhere to OpenAI's Usage Policies by informing end-users
that the voice they hear is AI-generated. Embrace the future of digital
audio with OpenAI!

# Addendum: Translating and Generating Audio in English and Spanish in Google Colab {#f3c6 .nc .nd .fr .be .ne .nf .ng .nh .ni .nj .nk .nl .nm .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .nz .bj selectable-paragraph=""}

# Preparing the English Text {#6bfa .nc .nd .fr .be .ne .nf .ng .nh .ni .nj .nk .nl .nm .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .nz .bj selectable-paragraph=""}

Store your English text in a variable named `my_speech`{.cw .pb .pc .pd
.ot .b}:

``` {.on .oo .op .oq .or .os .ot .ou .bo .ov .ba .bj}
my_speech = """
My name is Dr. Lee and I am a Data Analytics Professor at Miami Dade College. 
I am trying this ChatGPT model for the first time and it is super cool!
"""
```

# Translating the Text to Spanish Using OpenAI API {#5bbe .nc .nd .fr .be .ne .nf .ng .nh .ni .nj .nk .nl .nm .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .nz .bj selectable-paragraph=""}

Use the OpenAI API to translate `my_speech`{.cw .pb .pc .pd .ot .b} to
Spanish:

``` {.on .oo .op .oq .or .os .ot .ou .bo .ov .ba .bj}
# Ensure the OpenAI library is installed
!pip install --upgrade openai

import openai

# Translate the English text to Spanish
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Translate the following text to Spanish: {}".format(my_speech),
  max_tokens=100
)

# Extract the translated text
my_speech_es = response.choices[0].text.strip()
```

# Generating Audio in English and Spanish {#8b87 .nc .nd .fr .be .ne .nf .ng .nh .ni .nj .nk .nl .nm .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .nz .bj selectable-paragraph=""}

Generate audio files for both the original English text and its Spanish
translation.

``` {.on .oo .op .oq .or .os .ot .ou .bo .ov .ba .bj}
# Import the necessary library
from openai import OpenAI

client = OpenAI()

# Generate English audio
response_en = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=my_speech
)
response_en.stream_to_file("english_speech.mp3")

# Generate Spanish audio
response_es = client.audio.speech.create(
    model="tts-1",
    voice="alloy",  # Choose a voice suitable for Spanish
    input=my_speech_es
)
response_es.stream_to_file("spanish_speech.mp3")
```

# Playing the Audio Files {#88a5 .nc .nd .fr .be .ne .nf .ng .nh .ni .nj .nk .nl .nm .nn .no .np .nq .nr .ns .nt .nu .nv .nw .nx .ny .nz .bj selectable-paragraph=""}

Play the generated audio files using IPython's Audio class.

``` {.on .oo .op .oq .or .os .ot .ou .bo .ov .ba .bj}
from IPython.display import Audio

# Play the English audio
Audio("english_speech.mp3")

# Play the Spanish audio
Audio("spanish_speech.mp3")
```

This process allows you to translate text into Spanish using the OpenAI
API and then generate audio in both languages in a Google Colab
environment. It's a practical way to create bilingual audio content for
various applications, enhancing both accessibility and engagement for
diverse audiences.
