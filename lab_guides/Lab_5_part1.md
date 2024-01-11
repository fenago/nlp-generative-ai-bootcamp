

# Revolutionizing Higher Education with AI: Building Interactive Learning Assistants Using OpenAI APIs in Google Colab {#1de4 .pw-post-title .fp .fq .fr .be .fs .ft .fu .fv .fw .fx .fy .fz .ga .gb .gc .gd .ge .gf .gg .gh .gi .gj .gk .gl .gm .gn .go .gp .gq .gr .bj testid="storyTitle" selectable-paragraph=""}

::: {.gs .gt .gu .gv .gw}
::: {.speechify-ignore .ab .co}
::: {.speechify-ignore .bg .l}
::: {.gx .gy .gz .ha .hb .ab}
<div>

::: {.ab .hc}
[](https://drlee.io/?source=post_page-----a4991b0f2775--------------------------------){rel="noopener  ugc nofollow"}

<div>

::: {.bl aria-hidden="false" aria-describedby="1" aria-labelledby="1"}
::: {.l .hd .he .bx .hf .hg}
::: {.l .ee}
![Dr. Ernesto Lee](./Lab_2_files/1_nzdYUSs4c2RQs2W0FCHv1g.jpg){.l .eq
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
Lee](https://drlee.io/?source=post_page-----a4991b0f2775--------------------------------){.af
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
[7 min read]{testid="storyReadTime"}

::: {.ic .id .l aria-hidden="true"}
[[·]{.be .b .bf .z .dw}]{.l aria-hidden="true"}
:::

[Nov 24, 2023]{testid="storyPublishDate"}
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

::: {.bl aria-hidden="false" aria-describedby="72" aria-labelledby="72"}
![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdib3g9IjAgMCAyNCAyNCIgYXJpYS1sYWJlbD0iY2xhcCI+PHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0xMS4zNy44M0wxMiAzLjI4bC42My0yLjQ1aC0xLjI2ek0xMy45MiAzLjk1bDEuNTItMi4xLTEuMTgtLjQtLjM0IDIuNXpNOC41OSAxLjg0bDEuNTIgMi4xMS0uMzQtMi41LTEuMTguNHpNMTguNTIgMTguOTJhNC4yMyA0LjIzIDAgMCAxLTIuNjIgMS4zM2wuNDEtLjM3YzIuMzktMi40IDIuODYtNC45NSAxLjQtNy42M2wtLjkxLTEuNi0uOC0xLjY3Yy0uMjUtLjU2LS4xOS0uOTguMjEtMS4yOWEuNy43IDAgMCAxIC41NS0uMTNjLjI4LjA1LjU0LjIzLjcyLjVsMi4zNyA0LjE2Yy45NyAxLjYyIDEuMTQgNC4yMy0xLjMzIDYuN3ptLTExLS40NGwtNC4xNS00LjE1YS44My44MyAwIDAgMSAxLjE3LTEuMTdsMi4xNiAyLjE2YS4zNy4zNyAwIDAgMCAuNTEtLjUybC0yLjE1LTIuMTZMMy42IDExLjJhLjgzLjgzIDAgMCAxIDEuMTctMS4xN2wzLjQzIDMuNDRhLjM2LjM2IDAgMCAwIC41MiAwIC4zNi4zNiAwIDAgMCAwLS41Mkw1LjI5IDkuNTFsLS45Ny0uOTdhLjgzLjgzIDAgMCAxIDAtMS4xNi44NC44NCAwIDAgMSAxLjE3IDBsLjk3Ljk3IDMuNDQgMy40M2EuMzYuMzYgMCAwIDAgLjUxIDAgLjM3LjM3IDAgMCAwIDAtLjUyTDYuOTggNy44M2EuODIuODIgMCAwIDEtLjE4LS45LjgyLjgyIDAgMCAxIC43Ni0uNTFjLjIyIDAgLjQzLjA5LjU4LjI0bDUuOCA1Ljc5YS4zNy4zNyAwIDAgMCAuNTgtLjQyTDEzLjQgOS42N2MtLjI2LS41Ni0uMi0uOTguMi0xLjI5YS43LjcgMCAwIDEgLjU1LS4xM2MuMjguMDUuNTUuMjMuNzMuNWwyLjIgMy44NmMxLjMgMi4zOC44NyA0LjU5LTEuMjkgNi43NWE0LjY1IDQuNjUgMCAwIDEtNC4xOSAxLjM3IDcuNzMgNy43MyAwIDAgMS00LjA3LTIuMjV6bTMuMjMtMTIuNWwyLjEyIDIuMTFjLS40MS41LS40NyAxLjE3LS4xMyAxLjlsLjIyLjQ2LTMuNTItMy41M2EuODEuODEgMCAwIDEtLjEtLjM2YzAtLjIzLjA5LS40My4yNC0uNTlhLjg1Ljg1IDAgMCAxIDEuMTcgMHptNy4zNiAxLjdhMS44NiAxLjg2IDAgMCAwLTEuMjMtLjg0IDEuNDQgMS40NCAwIDAgMC0xLjEyLjI3Yy0uMy4yNC0uNS41NS0uNTguODktLjI1LS4yNS0uNTctLjQtLjkxLS40Ny0uMjgtLjA0LS41NiAwLS44Mi4xbC0yLjE4LTIuMThhMS41NiAxLjU2IDAgMCAwLTIuMiAwYy0uMi4yLS4zMy40NC0uNC43YTEuNTYgMS41NiAwIDAgMC0yLjYzLjc1IDEuNiAxLjYgMCAwIDAtMi4yMy0uMDQgMS41NiAxLjU2IDAgMCAwIDAgMi4yYy0uMjQuMS0uNS4yNC0uNzIuNDVhMS41NiAxLjU2IDAgMCAwIDAgMi4ybC41Mi41MmExLjU2IDEuNTYgMCAwIDAtLjc1IDIuNjFMNyAxOWE4LjQ2IDguNDYgMCAwIDAgNC40OCAyLjQ1IDUuMTggNS4xOCAwIDAgMCAzLjM2LS41IDQuODkgNC44OSAwIDAgMCA0LjItMS41MWMyLjc1LTIuNzcgMi41NC01Ljc0IDEuNDMtNy41OUwxOC4xIDcuNjh6Ij48L3BhdGg+PC9zdmc+)
:::

</div>

</div>
:::

::: {.pw-multi-vote-count .l .jz .ka .kb .kc .kd .ke .kf}
<div>

::: {.bl aria-hidden="false" aria-describedby="73" aria-labelledby="73"}
13[]{.l .h .g .f .qr .qs}
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

[](https://medium.com/plans?dimension=post_audio_button&postId=a4991b0f2775&source=upgrade_membership---post_audio_button----------------------------------){.af
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

::: {.bl aria-hidden="false" aria-describedby="78" aria-labelledby="78"}
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

![](./Lab_2_files/0_j0oQd8FXoIw9gLYf.jpg){.bg .kx .mc .c width="700"
height="700" loading="eager" role="presentation"}

# Unleashing the Power of AI in Learning Environments {#2803 .md .me .fr .be .mf .mg .mh .mi .mj .mk .ml .mm .mn .mo .mp .mq .mr .ms .mt .mu .mv .mw .mx .my .mz .na .bj selectable-paragraph=""}

In the swiftly advancing realm of education and corporate training,
Artificial Intelligence (AI) emerges as a transformative force. With its
unparalleled ability to process and analyze vast amounts of data, AI is
reshaping how knowledge is delivered and absorbed. This article delves
into the integration of OpenAI's advanced AI technologies in educational
settings, illustrating how educators and trainers can create AI
Assistants. These Assistants are not mere digital tools but intelligent
entities capable of enhancing learning experiences through personalized
and interactive engagement. We explore how AI can be a powerful ally in
the journey of knowledge impartation, making learning more adaptive,
engaging, and effective.

# The AI Advantage in Education and Training {#431d .md .me .fr .be .mf .mg .mh .mi .mj .mk .ml .mm .mn .mo .mp .mq .mr .ms .mt .mu .mv .mw .mx .my .mz .na .bj selectable-paragraph=""}

AI's integration into educational tools represents more than just a
technological upgrade; it's a paradigm shift in teaching and learning
methodologies. AI Assistants can offer instant, accurate, and
personalized responses to learners' queries, making education more
accessible and tailored to individual needs. In corporate training
scenarios, these Assistants can simulate real-world challenges, provide
situational analysis, and assist in skill development. The value lies in
AI's ability to create a more engaging, interactive, and adaptive
learning environment, bridging gaps in traditional educational methods.

# Setting Up: The First Steps in AI Integration {#57f7 .md .me .fr .be .mf .mg .mh .mi .mj .mk .ml .mm .mn .mo .mp .mq .mr .ms .mt .mu .mv .mw .mx .my .mz .na .bj selectable-paragraph=""}

Before diving into the complexities of AI, we must lay the groundwork.
This begins with setting up our coding environment to interact with
OpenAI's APIs.

## Code Breakdown: Environment Setup {#53ec .nz .me .fr .be .mf .oa .ob .oc .mj .od .oe .of .mn .nm .og .oh .oi .nq .oj .ok .ol .nu .om .on .oo .op .bj selectable-paragraph=""}

First, launch google colab and upload your documents. In my Dr. Lee GPT,
I uploaded two of my books: Natural Language Processing and Introduction
to Data Analytics:

![](./Lab_2_files/1_yzawjS9ufJjiubwM5m_AwA.png){.bg .kx .mc .c
width="700" height="318" loading="lazy" role="presentation"}

``` {.or .os .ot .ou .ov .ow .ox .oy .bo .oz .ba .bj}
!pip install --upgrade openai
import os

# Prompt for the API key
api_key = input("Enter your OpenAI API key: ")

# Set the environment variable
os.environ["OPENAI_API_KEY"] = api_key
from openai import OpenAI

client = OpenAI()
```

*Explanation*: Here, we're installing the OpenAI package and configuring
it with our unique API key. This key is crucial as it serves as our
access point to the OpenAI platform, enabling us to utilize its
extensive capabilities. The `client`{.cw .pl .pm .pn .ox .b} object
created is the primary tool through which we\'ll interact with the
OpenAI services.

# Creating a Knowledge Base: Uploading Course Materials {#af8d .md .me .fr .be .mf .mg .mh .mi .mj .mk .ml .mm .mn .mo .mp .mq .mr .ms .mt .mu .mv .mw .mx .my .mz .na .bj selectable-paragraph=""}

The effectiveness of an AI Assistant in an educational setting hinges on
its access to relevant, comprehensive knowledge. This is achieved by
uploading course or training materials to the OpenAI system.

## Code Breakdown: Uploading Files {#badd .nz .me .fr .be .mf .oa .ob .oc .mj .od .oe .of .mn .nm .og .oh .oi .nq .oj .ok .ol .nu .om .on .oo .op .bj selectable-paragraph=""}

``` {.or .os .ot .ou .ov .ow .ox .oy .bo .oz .ba .bj}
import os
import openai

# Directory containing the files
directory_path = '/content/drlee'

# List all files in the directory
all_file_names = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

# Initialize OpenAI client (ensure you have set your API key)
client = openai.OpenAI()

# Upload files and collect file IDs
file_ids = []
for file_name in all_file_names:
    with open(file_name, 'rb') as file:
        response = client.files.create(file=file, purpose='assistants')
        file_ids.append(response.id)  # Access the id attribute

# Now file_ids contains the IDs of all uploaded files
```

*Explanation*: This segment of code is about building the AI's database.
By iterating over a directory of educational materials and uploading
each file to OpenAI, we create a repository of knowledge that the AI can
draw from. Each file's ID is stored, linking the content directly to our
upcoming AI Assistant.

# Bringing the AI Assistant to Life {#8198 .md .me .fr .be .mf .mg .mh .mi .mj .mk .ml .mm .mn .mo .mp .mq .mr .ms .mt .mu .mv .mw .mx .my .mz .na .bj selectable-paragraph=""}

With our knowledge base ready, the next step is to create the AI
Assistant itself. This involves configuring the Assistant's capabilities
to ensure it can effectively utilize the knowledge we've provided.

## Code Breakdown: Creating the AI Assistant {#54e8 .nz .me .fr .be .mf .oa .ob .oc .mj .od .oe .of .mn .nm .og .oh .oi .nq .oj .ok .ol .nu .om .on .oo .op .bj selectable-paragraph=""}

``` {.or .os .ot .ou .ov .ow .ox .oy .bo .oz .ba .bj}
# Create the Assistant
assistant2 = client.beta.assistants.create(
    name="Academic Co-Pilot",
    instructions="You are an AI assistant tailored to a Professor's expertise in [Subject Area]: AI and Data Analytics. Provide detailed and accurate information based on the provided resources ONLY.  If there is no context in the resources then provide a blank response",
    tools=[{"type": "retrieval"}],  # You can specify the 'code_interpreter' tool if needed
    model="gpt-4-1106-preview",
    file_ids=file_ids  # Assuming 'file_ids' is a list of IDs from previously uploaded files
)
```

Here is another option for the assistant:

``` {.or .os .ot .ou .ov .ow .ox .oy .bo .oz .ba .bj}
# Create the Assistant with both Retrieval and Code Interpreter tools
assistant = client.beta.assistants.create(
    name="Academic Co-Pilot",
    instructions="You are an AI assistant tailored to a Professor's expertise in [Subject Area]: AI and Data Analytics. Provide detailed and accurate information based on the provided resources ONLY.  If there is no context in the resources then provide a blank response.  Please provide citations for the documents and page numbers",
    tools=[
        {"type": "retrieval"},
        {"type": "code_interpreter"}
    ],
    model="gpt-4-1106-preview",
    file_ids=file_ids  # Assuming 'file_ids' is a list of IDs from previously uploaded files
)
```

*Explanation*: This code initiates the creation of our AI Assistant.
Named "Educational Assistant," it is equipped with a retrieval tool,
allowing it to search and present information from the uploaded
documents. The use of OpenAI's GPT-4 model ensures that the responses
are not only relevant but also contextually nuanced, making the
Assistant a valuable tool for educational purposes.

# Interactive Learning: Engaging with the AI {#c66d .md .me .fr .be .mf .mg .mh .mi .mj .mk .ml .mm .mn .mo .mp .mq .mr .ms .mt .mu .mv .mw .mx .my .mz .na .bj selectable-paragraph=""}

The real power of an AI Assistant in education lies in its ability to
interact and respond to queries. This is where we set up a system for
engaging in a dialogue with the Assistant.

## Code Breakdown: Interactive Dialogue with AI {#9da5 .nz .me .fr .be .mf .oa .ob .oc .mj .od .oe .of .mn .nm .og .oh .oi .nq .oj .ok .ol .nu .om .on .oo .op .bj selectable-paragraph=""}

thread will create a conversation with the AI.

``` {.or .os .ot .ou .ov .ow .ox .oy .bo .oz .ba .bj}
thread = client.beta.threads.create()
```

Now that thread represents the conversation, let's initiate the
conversation:

``` {.or .os .ot .ou .ov .ow .ox .oy .bo .oz .ba .bj}
user_query = "What are stop words?"
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content=user_query
)
print(message)
```

![](./Lab_2_files/1_EMr9XpTmFH3ilpVsRTmz1g.png){.bg .kx .mc .c
width="700" height="200" loading="lazy" role="presentation"}

Finally, let's kick off the conversation so that it actually starts:

``` {.or .os .ot .ou .ov .ow .ox .oy .bo .oz .ba .bj}
run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id,
  instructions='''Please address the user as MDC Student X. If there is no context for the answer in the documents then don't return anything'''
)
```

Now that the conversation has been sent to the API, we must wait for it
to response. The best way is to put a timer on the response and just
keep checking to see when it is finished:

``` {.or .os .ot .ou .ov .ow .ox .oy .bo .oz .ba .bj}
import time

while True:
    # Retrieve the run's current status
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    print(run)

    # Check if the run has a 'completed_at' value
    if run.completed_at is not None:
        break

    # Wait for 7 seconds before checking again
    time.sleep(7)

# Continue with the next steps after the loop exits
```

![](./Lab_2_files/1_tK8Ro6-msRai5z5neGLdFQ.png){.bg .kx .mc .c
width="700" height="252" loading="lazy" role="presentation"}

When it finally ends, we can view the results:

``` {.or .os .ot .ou .ov .ow .ox .oy .bo .oz .ba .bj}
messages = client.beta.threads.messages.list(
  thread_id=thread.id
)
print(messages)
```

![](./Lab_2_files/1_7fQNTCfi3_jq0ArtG1xk0A.png){.bg .kx .mc .c
width="700" height="90" loading="lazy" role="presentation"}

Let's look at this in a clean and formatted manner:

``` {.or .os .ot .ou .ov .ow .ox .oy .bo .oz .ba .bj}
import textwrap

# Formatting and displaying the results
for message in messages.data:
    # Extracting the message content
    if message.content and message.content[0].text.value.strip():
        content = message.content[0].text.value
    else:
        content = "There is no context for this question."

    # Wrap text to the width of the colab cell
    wrapped_content = textwrap.fill(content, width=80)
    
    # Printing the role and content in a readable format
    print(f"Role: {message.role}")
    print("Content:")
    print(wrapped_content)
    print("-" * 80)  # Separator for readability
```

![](./Lab_2_files/1_SAivIYN7KweMMSi-kyG9Lw.png){.bg .kx .mc .c
width="617" height="569" loading="lazy" role="presentation"}

*Explanation*: In this section, we establish a 'thread' --- a continuous
stream of interaction between the user and the AI. The user's query is
sent to the AI, and the AI, utilizing its knowledge base and the
instructions provided, generates a response. This interactive feature
simulates a real-world educational dialogue, enhancing the learning
experience.

# Conclusion: AI as a Catalyst in Education {#4768 .md .me .fr .be .mf .mg .mh .mi .mj .mk .ml .mm .mn .mo .mp .mq .mr .ms .mt .mu .mv .mw .mx .my .mz .na .bj selectable-paragraph=""}

The integration of AI into educational and training programs represents
a significant leap forward. By creating an AI Assistant with OpenAI's
APIs, educators and trainers can offer a more personalized, interactive,
and effective learning experience. This technology is not about
replacing traditional methods but enhancing and supplementing them,
opening new avenues for engagement and understanding.

As we stand at the cusp of a new era in education and corporate
training, the integration of AI through platforms like OpenAI is not
just an advancement; it's a revolutionary shift. The creation and
implementation of AI Assistants represent a significant leap in the way
knowledge is imparted and received. These Assistants, powered by the
sophisticated algorithms of OpenAI, bring a dimension of
personalization, interactivity, and adaptability that traditional
methods alone cannot offer.

The value of such AI integration in educational settings is manifold.
For students and learners, AI Assistants provide instant access to
information, tailored responses to queries, and an interactive learning
experience that adapts to their individual pace and style of learning.
This personalized approach can effectively address diverse learning
needs and styles, fostering a more inclusive and effective educational
environment.

In corporate training scenarios, AI Assistants can simulate real-world
challenges, offering trainees the opportunity to apply concepts in
practical settings. This not only enhances skill development but also
prepares employees for the complexities of the modern workplace. The
ability of AI to provide instant feedback and nuanced insights can
accelerate the learning process, making training more efficient and
impactful.

Moreover, the integration of AI in education and training paves the way
for continuous innovation. As AI technology evolves, so too will the
capabilities of these Assistants, continuously enriching the learning
experience. Educators and trainers are empowered to update and adapt
their teaching materials dynamically, ensuring that their content
remains relevant and effective.

In conclusion, the utilization of OpenAI's APIs to create AI Assistants
is a testament to the transformative power of AI in the realm of
education and training. It's a step towards a future where learning is
more dynamic, responsive, and aligned with the evolving needs of
learners and the demands of a rapidly changing world. As we embrace
these technological advancements, we open up new possibilities for
growth, understanding, and innovation in the field of education. The
journey into AI-assisted learning is not just about embracing new
technology; it's about opening doors to a world of limitless educational
possibilities.